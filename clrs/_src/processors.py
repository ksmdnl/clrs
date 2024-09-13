# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""JAX implementation of baseline processor networks."""

import abc
from typing import Any, Callable, List, Optional, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


_Array = chex.Array
_Fn = Callable[..., Any]
BIG_NUMBER = 1e6
PROCESSOR_TAG = 'clrs_processor'


class Processor(hk.Module):
  """Processor abstract base class."""

  def __init__(self, name: str):
    if not name.endswith(PROCESSOR_TAG):
      name = name + '_' + PROCESSOR_TAG
    super().__init__(name=name)

  @abc.abstractmethod
  def __call__(
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **kwargs,
  ) -> Tuple[_Array, Optional[_Array]]:
    """Processor inference step.

    Args:
      node_fts: Node features.
      edge_fts: Edge features.
      graph_fts: Graph features.
      adj_mat: Graph adjacency matrix.
      hidden: Hidden features.
      **kwargs: Extra kwargs.

    Returns:
      Output of processor inference step as a 2-tuple of (node, edge)
      embeddings. The edge embeddings can be None.
    """
    pass

  @property
  def inf_bias(self):
    return False

  @property
  def inf_bias_edge(self):
    return False

class MLP(hk.Module):
  def __init__(
      self, input_dim, output_dim, dropout: float = 0.0, linear: bool = False
  ):
    super().__init__(name="MLP")
    self.linear = linear
    self.output_dim = output_dim
    self.hidden_dim = output_dim
    self.dropout_rate = dropout

  def __call__(self, x, is_training=False):
    if self.linear:
      return hk.Linear(self.output_dim)(x)
    else:
      x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
      x = hk.Linear(self.hidden_dim)(x)
      if is_training:
        x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
      x = jax.nn.relu(x)
      x = hk.Linear(self.output_dim)(x)
      if is_training:
        x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
    return x

class EdgeAttention(hk.Module):
  def __init__(self, d_model, num_heads, dropout):
    super().__init__(name="EdgeAttention")
    self.rate = dropout 
    self.d_model = d_model
    self.num_heads = num_heads
    self.d_k = d_model // num_heads

  def __call__(self, query, key, value, mask=None, is_training=False):
    linears = [hk.Linear(self.d_model, with_bias=False) for _ in range(5)]
    num_batches = query.shape[0]
    num_nodes_q = query.shape[1]
    num_nodes_k = key.shape[1]

    left_k, right_k, left_v, right_v = [
        l(x) for l, x in zip(linears, (query, key, value, value))
    ]
    left_k = left_k.reshape(
        (num_batches, num_nodes_q, num_nodes_q, self.num_heads, self.d_k)
    )
    right_k = right_k.reshape(
        (num_batches, key.shape[1], key.shape[2], self.num_heads, self.d_k)
    )
    left_v = left_v.reshape(right_k.shape)
    right_v = right_v.reshape(right_k.shape)

    scores = jnp.einsum("bxahd,bayhd->bxayh", left_k, right_k) / jnp.sqrt(
        self.d_k
    )

    if mask is not None:
        scores_dtype = scores.dtype
        scores = scores.astype(jnp.float32).at[..., None].set(mask, -1e9).astype(scores_dtype)

    att = jax.nn.softmax(scores, axis=2)
    if is_training:
      att = hk.dropout(hk.next_rng_key(), self.rate, att)

    val = jnp.einsum("bxahd,bayhd->bxayhd", left_v, right_v)
    x = jnp.einsum("bxayh,bxayhd->bxyhd", att, val)
    x = x.reshape((num_batches, num_nodes_q, num_nodes_k, self.d_model))

    return linears[-1](x)


class FFN(hk.Module):
  def __init__(
      self,
      embed_dim: int,
      dropout: float = 0,
      activation: str = "relu",
      norm: str = "layer",
  ):
    super().__init__(name="FFN")

    self.activation = activation
    self.embed_dim = embed_dim
    self.rate = dropout
    self.norm = norm
    self.embed_dim = embed_dim

  def __call__(self, x_prior, x, rng=None, is_training=False):
    if self.activation == "relu":
        activation_fn = jax.nn.relu
    elif self.activation == "gelu":
        activation_fn = jax.nn.gelu
    else:
        raise ValueError(f"Activation function {self.activation} is not supported")

    if self.norm == "layer":
        norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        norm_aggregate = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    else:
        raise ValueError(f"Norm {norm} is not supported")

    if is_training:
      x = hk.dropout(hk.next_rng_key(), self.rate, x)
    x = x_prior + x
    x = norm_aggregate(x)

    x_intermediate = activation_fn(hk.Linear(self.embed_dim * 2)(x))
    if is_training:
      x_intermediate = hk.dropout(hk.next_rng_key(), self.rate, x_intermediate)
    x_intermediate = hk.Linear(self.embed_dim)(x_intermediate)
    if is_training:
      x_intermediate = hk.dropout(hk.next_rng_key(), self.rate, x_intermediate)

    x = x_intermediate + x
    return norm(x)

class EdgeTransformerLayer(hk.Module):
  def __init__(
      self,
      embed_dim: int,
      num_heads: int,
      dropout: float,
      attention_dropout: float,
      activation: str = 'relu',
      norm_first: bool = False,
  ):
    super().__init__(name="ET_Layer")
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.dropout = dropout
    self.attention_dropout = attention_dropout
    self.activation = activation
    self.norm_first = norm_first

  def __call__(self, x_in, mask=None, is_training=False):
    attention = EdgeAttention(self.embed_dim, self.num_heads, self.attention_dropout)
    ffn = FFN(self.embed_dim, self.dropout, self.activation, 'layer')
    x = x_in

    if self.norm_first:
      x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)

    if mask is not None:
        x_upd = attention(x, x, x, ~mask, is_training=is_training)
    else:
        x_upd = attention(x, x, x, is_training=is_training)

    x = ffn(x_in, x_upd, is_training=is_training)
    return x

class EdgeTransformer(hk.Module):
  def __init__(
      self,
      out_size: int,
      nb_heads: int,
      num_layers: int = 5,
      dropout: float = 0.0,
      attention_dropout: float = 0.0,
      residual: bool = False,
      use_ln: bool = False,
      activation: str = 'relu',
      norm_first: bool = False,
  ):
    super().__init__(name="EdgeTransformer")
    self.out_size = out_size
    self.nb_heads = nb_heads
    self.dropout = dropout
    self.attention_dropout = attention_dropout
    self.residual = residual
    self.use_ln = use_ln
    self.num_layers = num_layers
    self.activation = activation
    self.norm_first = norm_first

    if out_size % nb_heads != 0:
      raise ValueError('The number of attention heads must divide the width!')

  def __call__(self, x, is_training=False):
    """EdgeTransformer inference step."""
    for _ in range(self.num_layers):
      x = EdgeTransformerLayer(embed_dim=self.out_size,
            num_heads=self.nb_heads,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            activation=self.activation,
            norm_first=self.norm_first)(x, is_training=is_training)
    return x

class ET_Processor(Processor):
  def __init__(
      self,
      out_size: int,
      nb_heads: int,
      num_layers: int,
      activation: str = 'relu',
      dropout: float = 0.0,
      attention_dropout: float = 0.0,
      residual: bool = False,
      use_ln: bool = False,
      name: str = 'edge_t',
      norm_first: bool = False,
  ):
    super().__init__(name=name)

    self.concat_dim = 2 * out_size
    self.out_size = out_size
    self.nb_heads = nb_heads
    self.dropout = dropout
    self.attention_dropout = attention_dropout
    self.num_layers = num_layers
    self.activation = activation
    self.norm_first = norm_first

  def __call__(
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **kwargs,
  ) -> Tuple[_Array, Optional[_Array]]:
    """EdgeTransformer inference step."""
    node_proj = hk.Linear(self.out_size)

    transformer = EdgeTransformer(
        out_size=self.out_size,
        nb_heads=self.nb_heads,
        dropout=self.dropout,
        attention_dropout=self.attention_dropout,
        num_layers=self.num_layers,
        activation=self.activation,
        norm_first=self.norm_first,
    )

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)
    is_training = not kwargs['repred']
    readout = kwargs['readout']
    is_graph_fts_avail = kwargs["is_graph_fts_avail"]

    node_fts = jnp.concatenate([node_fts, hidden], axis=-1)
    node_proj = hk.Linear(self.out_size)

    if is_graph_fts_avail:
      graph_fts = jnp.tile(jnp.expand_dims(graph_fts, -2), (1, n, 1))
      node_fts = jnp.concatenate([node_fts, graph_fts], axis=-1)

    """Composer"""
    z = jnp.expand_dims(node_fts, axis=2)  # Shape: (b, n, 1, d)
    z_repeated = jnp.tile(z, reps=(1, 1, n, 1))  # Shape: (b, n, n, d)
    z_repeated_t = jnp.transpose(z_repeated, (0, 2, 1, 3))
    B = jnp.concatenate([z_repeated, z_repeated_t], axis=3)  # Shape: (b, n, n, 2d)

    proj_node_fts = node_proj(B)
    z_composed = proj_node_fts + edge_fts

    """EdgeTransformer"""
    z_out = transformer(z_composed, is_training=is_training)

    if readout == 'diagonal':
      z_out_diag = jnp.einsum('biid->bid', z_out) # Shape: (b, n, d)
    else:
      b, n, d = node_fts.shape
      linear_layer = hk.Linear(2 * d)
      B_linear = linear_layer(z_out)

      B_linear_first_d = B_linear[:, :, :, :d]
      B_linear_other_d = B_linear[:, :, :, d:]

      B_linear_first_d_sum = jnp.sum(B_linear_first_d, axis=1)  # Shape: (b, n, d)
      B_linear_other_d_sum = jnp.sum(B_linear_other_d, axis=2)  # Shape: (b, n, d)

      z_out_diag = B_linear_first_d_sum + B_linear_other_d_sum  # Shape: (b, n, d)
      z_out_diag = MLP(d, self.out_size, linear=False)(z_out_diag, is_training=is_training)

    return z_out_diag, z_out

class GAT(Processor):
  """Graph Attention Network (Velickovic et al., ICLR 2018)."""

  def __init__(
      self,
      out_size: int,
      nb_heads: int,
      activation: Optional[_Fn] = jax.nn.relu,
      residual: bool = True,
      use_ln: bool = False,
      name: str = 'gat_aggr',
  ):
    super().__init__(name=name)
    self.out_size = out_size
    self.nb_heads = nb_heads
    if out_size % nb_heads != 0:
      raise ValueError('The number of attention heads must divide the width!')
    self.head_size = out_size // nb_heads
    self.activation = activation
    self.residual = residual
    self.use_ln = use_ln

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """GAT inference step."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    z = jnp.concatenate([node_fts, hidden], axis=-1)
    m = hk.Linear(self.out_size)
    skip = hk.Linear(self.out_size)

    bias_mat = (adj_mat - 1.0) * 1e9
    bias_mat = jnp.tile(bias_mat[..., None],
                        (1, 1, 1, self.nb_heads))     # [B, N, N, H]
    bias_mat = jnp.transpose(bias_mat, (0, 3, 1, 2))  # [B, H, N, N]

    a_1 = hk.Linear(self.nb_heads)
    a_2 = hk.Linear(self.nb_heads)
    a_e = hk.Linear(self.nb_heads)
    a_g = hk.Linear(self.nb_heads)

    values = m(z)                                      # [B, N, H*F]
    values = jnp.reshape(
        values,
        values.shape[:-1] + (self.nb_heads, self.head_size))  # [B, N, H, F]
    values = jnp.transpose(values, (0, 2, 1, 3))              # [B, H, N, F]

    att_1 = jnp.expand_dims(a_1(z), axis=-1)
    att_2 = jnp.expand_dims(a_2(z), axis=-1)
    att_e = a_e(edge_fts)
    att_g = jnp.expand_dims(a_g(graph_fts), axis=-1)

    logits = (
        jnp.transpose(att_1, (0, 2, 1, 3)) +  # + [B, H, N, 1]
        jnp.transpose(att_2, (0, 2, 3, 1)) +  # + [B, H, 1, N]
        jnp.transpose(att_e, (0, 3, 1, 2)) +  # + [B, H, N, N]
        jnp.expand_dims(att_g, axis=-1)       # + [B, H, 1, 1]
    )                                         # = [B, H, N, N]
    coefs = jax.nn.softmax(jax.nn.leaky_relu(logits) + bias_mat, axis=-1)
    ret = jnp.matmul(coefs, values)  # [B, H, N, F]
    ret = jnp.transpose(ret, (0, 2, 1, 3))  # [B, N, H, F]
    ret = jnp.reshape(ret, ret.shape[:-2] + (self.out_size,))  # [B, N, H*F]

    if self.residual:
      ret += skip(z)

    if self.activation is not None:
      ret = self.activation(ret)

    if self.use_ln:
      ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ret = ln(ret)

    return ret, None  # pytype: disable=bad-return-type  # numpy-scalars


class GATFull(GAT):
  """Graph Attention Network with full adjacency matrix."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class GATv2(Processor):
  """Graph Attention Network v2 (Brody et al., ICLR 2022)."""

  def __init__(
      self,
      out_size: int,
      nb_heads: int,
      mid_size: Optional[int] = None,
      activation: Optional[_Fn] = jax.nn.relu,
      residual: bool = True,
      use_ln: bool = False,
      name: str = 'gatv2_aggr',
  ):
    super().__init__(name=name)
    if mid_size is None:
      self.mid_size = out_size
    else:
      self.mid_size = mid_size
    self.out_size = out_size
    self.nb_heads = nb_heads
    if out_size % nb_heads != 0:
      raise ValueError('The number of attention heads must divide the width!')
    self.head_size = out_size // nb_heads
    if self.mid_size % nb_heads != 0:
      raise ValueError('The number of attention heads must divide the message!')
    self.mid_head_size = self.mid_size // nb_heads
    self.activation = activation
    self.residual = residual
    self.use_ln = use_ln

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """GATv2 inference step."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    z = jnp.concatenate([node_fts, hidden], axis=-1)
    m = hk.Linear(self.out_size)
    skip = hk.Linear(self.out_size)

    bias_mat = (adj_mat - 1.0) * 1e9
    bias_mat = jnp.tile(bias_mat[..., None],
                        (1, 1, 1, self.nb_heads))     # [B, N, N, H]
    bias_mat = jnp.transpose(bias_mat, (0, 3, 1, 2))  # [B, H, N, N]

    w_1 = hk.Linear(self.mid_size)
    w_2 = hk.Linear(self.mid_size)
    w_e = hk.Linear(self.mid_size)
    w_g = hk.Linear(self.mid_size)

    a_heads = []
    for _ in range(self.nb_heads):
      a_heads.append(hk.Linear(1))

    values = m(z)                                      # [B, N, H*F]
    values = jnp.reshape(
        values,
        values.shape[:-1] + (self.nb_heads, self.head_size))  # [B, N, H, F]
    values = jnp.transpose(values, (0, 2, 1, 3))              # [B, H, N, F]

    pre_att_1 = w_1(z)
    pre_att_2 = w_2(z)
    pre_att_e = w_e(edge_fts)
    pre_att_g = w_g(graph_fts)

    pre_att = (
        jnp.expand_dims(pre_att_1, axis=1) +     # + [B, 1, N, H*F]
        jnp.expand_dims(pre_att_2, axis=2) +     # + [B, N, 1, H*F]
        pre_att_e +                              # + [B, N, N, H*F]
        jnp.expand_dims(pre_att_g, axis=(1, 2))  # + [B, 1, 1, H*F]
    )                                            # = [B, N, N, H*F]

    pre_att = jnp.reshape(
        pre_att,
        pre_att.shape[:-1] + (self.nb_heads, self.mid_head_size)
    )  # [B, N, N, H, F]

    pre_att = jnp.transpose(pre_att, (0, 3, 1, 2, 4))  # [B, H, N, N, F]

    # This part is not very efficient, but we agree to keep it this way to
    # enhance readability, assuming `nb_heads` will not be large.
    logit_heads = []
    for head in range(self.nb_heads):
      logit_heads.append(
          jnp.squeeze(
              a_heads[head](jax.nn.leaky_relu(pre_att[:, head])),
              axis=-1)
      )  # [B, N, N]

    logits = jnp.stack(logit_heads, axis=1)  # [B, H, N, N]

    coefs = jax.nn.softmax(logits + bias_mat, axis=-1)
    ret = jnp.matmul(coefs, values)  # [B, H, N, F]
    ret = jnp.transpose(ret, (0, 2, 1, 3))  # [B, N, H, F]
    ret = jnp.reshape(ret, ret.shape[:-2] + (self.out_size,))  # [B, N, H*F]

    if self.residual:
      ret += skip(z)

    if self.activation is not None:
      ret = self.activation(ret)

    if self.use_ln:
      ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ret = ln(ret)

    return ret, None  # pytype: disable=bad-return-type  # numpy-scalars


class GATv2FullD2(GATv2):
  """Graph Attention Network v2 with full adjacency matrix and D2 symmetry."""

  def d2_forward(self,
                 node_fts: List[_Array],
                 edge_fts: List[_Array],
                 graph_fts: List[_Array],
                 adj_mat: _Array,
                 hidden: _Array,
                 **unused_kwargs) -> List[_Array]:
    num_d2_actions = 4

    d2_inverses = [
        0, 1, 2, 3  # All members of D_2 are self-inverses!
    ]

    d2_multiply = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ]

    assert len(node_fts) == num_d2_actions
    assert len(edge_fts) == num_d2_actions
    assert len(graph_fts) == num_d2_actions

    ret_nodes = []
    adj_mat = jnp.ones_like(adj_mat)

    for g in range(num_d2_actions):
      emb_values = []
      for h in range(num_d2_actions):
        gh = d2_multiply[d2_inverses[g]][h]
        node_features = jnp.concatenate(
            (node_fts[g], node_fts[gh]),
            axis=-1)
        edge_features = jnp.concatenate(
            (edge_fts[g], edge_fts[gh]),
            axis=-1)
        graph_features = jnp.concatenate(
            (graph_fts[g], graph_fts[gh]),
            axis=-1)
        cell_embedding = super().__call__(
            node_fts=node_features,
            edge_fts=edge_features,
            graph_fts=graph_features,
            adj_mat=adj_mat,
            hidden=hidden
        )
        emb_values.append(cell_embedding[0])
      ret_nodes.append(
          jnp.mean(jnp.stack(emb_values, axis=0), axis=0)
      )

    return ret_nodes


class GATv2Full(GATv2):
  """Graph Attention Network v2 with full adjacency matrix."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


def get_triplet_msgs(z, edge_fts, graph_fts, nb_triplet_fts):
  """Triplet messages, as done by Dudzik and Velickovic (2022)."""
  t_1 = hk.Linear(nb_triplet_fts)
  t_2 = hk.Linear(nb_triplet_fts)
  t_3 = hk.Linear(nb_triplet_fts)
  t_e_1 = hk.Linear(nb_triplet_fts)
  t_e_2 = hk.Linear(nb_triplet_fts)
  t_e_3 = hk.Linear(nb_triplet_fts)
  t_g = hk.Linear(nb_triplet_fts)

  tri_1 = t_1(z)
  tri_2 = t_2(z)
  tri_3 = t_3(z)
  tri_e_1 = t_e_1(edge_fts)
  tri_e_2 = t_e_2(edge_fts)
  tri_e_3 = t_e_3(edge_fts)
  tri_g = t_g(graph_fts)

  return (
      jnp.expand_dims(tri_1, axis=(2, 3))    +  #   (B, N, 1, 1, H)
      jnp.expand_dims(tri_2, axis=(1, 3))    +  # + (B, 1, N, 1, H)
      jnp.expand_dims(tri_3, axis=(1, 2))    +  # + (B, 1, 1, N, H)
      jnp.expand_dims(tri_e_1, axis=3)       +  # + (B, N, N, 1, H)
      jnp.expand_dims(tri_e_2, axis=2)       +  # + (B, N, 1, N, H)
      jnp.expand_dims(tri_e_3, axis=1)       +  # + (B, 1, N, N, H)
      jnp.expand_dims(tri_g, axis=(1, 2, 3))    # + (B, 1, 1, 1, H)
  )                                             # = (B, N, N, N, H)


class PGN(Processor):
  """Pointer Graph Networks (Veličković et al., NeurIPS 2020)."""

  def __init__(
      self,
      out_size: int,
      mid_size: Optional[int] = None,
      mid_act: Optional[_Fn] = None,
      activation: Optional[_Fn] = jax.nn.relu,
      reduction: _Fn = jnp.max,
      msgs_mlp_sizes: Optional[List[int]] = None,
      use_ln: bool = False,
      use_triplets: bool = False,
      nb_triplet_fts: int = 8,
      gated: bool = False,
      name: str = 'mpnn_aggr',
  ):
    super().__init__(name=name)
    if mid_size is None:
      self.mid_size = out_size
    else:
      self.mid_size = mid_size
    self.out_size = out_size
    self.mid_act = mid_act
    self.activation = activation
    self.reduction = reduction
    self._msgs_mlp_sizes = msgs_mlp_sizes
    self.use_ln = use_ln
    self.use_triplets = use_triplets
    self.nb_triplet_fts = nb_triplet_fts
    self.gated = gated

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """MPNN inference step."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    z = jnp.concatenate([node_fts, hidden], axis=-1)
    m_1 = hk.Linear(self.mid_size)
    m_2 = hk.Linear(self.mid_size)
    m_e = hk.Linear(self.mid_size)
    m_g = hk.Linear(self.mid_size)

    o1 = hk.Linear(self.out_size)
    o2 = hk.Linear(self.out_size)

    msg_1 = m_1(z)
    msg_2 = m_2(z)
    msg_e = m_e(edge_fts)
    msg_g = m_g(graph_fts)

    tri_msgs = None

    if self.use_triplets:
      # Triplet messages, as done by Dudzik and Velickovic (2022)
      triplets = get_triplet_msgs(z, edge_fts, graph_fts, self.nb_triplet_fts)

      o3 = hk.Linear(self.out_size)
      tri_msgs = o3(jnp.max(triplets, axis=1))  # (B, N, N, H)

      if self.activation is not None:
        tri_msgs = self.activation(tri_msgs)

    msgs = (
        jnp.expand_dims(msg_1, axis=1) + jnp.expand_dims(msg_2, axis=2) +
        msg_e + jnp.expand_dims(msg_g, axis=(1, 2)))

    if self._msgs_mlp_sizes is not None:
      msgs = hk.nets.MLP(self._msgs_mlp_sizes)(jax.nn.relu(msgs))

    if self.mid_act is not None:
      msgs = self.mid_act(msgs)

    if self.reduction == jnp.mean:
      msgs = jnp.sum(msgs * jnp.expand_dims(adj_mat, -1), axis=1)
      msgs = msgs / jnp.sum(adj_mat, axis=-1, keepdims=True)
    elif self.reduction == jnp.max:
      maxarg = jnp.where(jnp.expand_dims(adj_mat, -1),
                         msgs,
                         -BIG_NUMBER)
      msgs = jnp.max(maxarg, axis=1)
    else:
      msgs = self.reduction(msgs * jnp.expand_dims(adj_mat, -1), axis=1)

    h_1 = o1(z)
    h_2 = o2(msgs)

    ret = h_1 + h_2

    if self.activation is not None:
      ret = self.activation(ret)

    if self.use_ln:
      ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ret = ln(ret)

    if self.gated:
      gate1 = hk.Linear(self.out_size)
      gate2 = hk.Linear(self.out_size)
      gate3 = hk.Linear(self.out_size, b_init=hk.initializers.Constant(-3))
      gate = jax.nn.sigmoid(gate3(jax.nn.relu(gate1(z) + gate2(msgs))))
      ret = ret * gate + hidden * (1-gate)

    return ret, tri_msgs  # pytype: disable=bad-return-type  # numpy-scalars


class DeepSets(PGN):
  """Deep Sets (Zaheer et al., NeurIPS 2017)."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    assert adj_mat.ndim == 3
    adj_mat = jnp.ones_like(adj_mat) * jnp.eye(adj_mat.shape[-1])
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class MPNN(PGN):
  """Message-Passing Neural Network (Gilmer et al., ICML 2017)."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class PGNMask(PGN):
  """Masked Pointer Graph Networks (Veličković et al., NeurIPS 2020)."""

  @property
  def inf_bias(self):
    return True

  @property
  def inf_bias_edge(self):
    return True


class MemNetMasked(Processor):
  """Implementation of End-to-End Memory Networks.

  Inspired by the description in https://arxiv.org/abs/1503.08895.
  """

  def __init__(
      self,
      vocab_size: int,
      sentence_size: int,
      linear_output_size: int,
      embedding_size: int = 16,
      memory_size: Optional[int] = 128,
      num_hops: int = 1,
      nonlin: Callable[[Any], Any] = jax.nn.relu,
      apply_embeddings: bool = True,
      init_func: hk.initializers.Initializer = jnp.zeros,
      use_ln: bool = False,
      name: str = 'memnet') -> None:
    """Constructor.

    Args:
      vocab_size: the number of words in the dictionary (each story, query and
        answer come contain symbols coming from this dictionary).
      sentence_size: the dimensionality of each memory.
      linear_output_size: the dimensionality of the output of the last layer
        of the model.
      embedding_size: the dimensionality of the latent space to where all
        memories are projected.
      memory_size: the number of memories provided.
      num_hops: the number of layers in the model.
      nonlin: non-linear transformation applied at the end of each layer.
      apply_embeddings: flag whether to aply embeddings.
      init_func: initialization function for the biases.
      use_ln: whether to use layer normalisation in the model.
      name: the name of the model.
    """
    super().__init__(name=name)
    self._vocab_size = vocab_size
    self._embedding_size = embedding_size
    self._sentence_size = sentence_size
    self._memory_size = memory_size
    self._linear_output_size = linear_output_size
    self._num_hops = num_hops
    self._nonlin = nonlin
    self._apply_embeddings = apply_embeddings
    self._init_func = init_func
    self._use_ln = use_ln
    # Encoding part: i.e. "I" of the paper.
    self._encodings = _position_encoding(sentence_size, embedding_size)

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """MemNet inference step."""

    del hidden
    node_and_graph_fts = jnp.concatenate([node_fts, graph_fts[:, None]],
                                         axis=1)
    edge_fts_padded = jnp.pad(edge_fts * adj_mat[..., None],
                              ((0, 0), (0, 1), (0, 1), (0, 0)))
    nxt_hidden = jax.vmap(self._apply, (1), 1)(node_and_graph_fts,
                                               edge_fts_padded)

    # Broadcast hidden state corresponding to graph features across the nodes.
    nxt_hidden = nxt_hidden[:, :-1] + nxt_hidden[:, -1:]
    return nxt_hidden, None  # pytype: disable=bad-return-type  # numpy-scalars

  def _apply(self, queries: _Array, stories: _Array) -> _Array:
    """Apply Memory Network to the queries and stories.

    Args:
      queries: Tensor of shape [batch_size, sentence_size].
      stories: Tensor of shape [batch_size, memory_size, sentence_size].

    Returns:
      Tensor of shape [batch_size, vocab_size].
    """
    if self._apply_embeddings:
      query_biases = hk.get_parameter(
          'query_biases',
          shape=[self._vocab_size - 1, self._embedding_size],
          init=self._init_func)
      stories_biases = hk.get_parameter(
          'stories_biases',
          shape=[self._vocab_size - 1, self._embedding_size],
          init=self._init_func)
      memory_biases = hk.get_parameter(
          'memory_contents',
          shape=[self._memory_size, self._embedding_size],
          init=self._init_func)
      output_biases = hk.get_parameter(
          'output_biases',
          shape=[self._vocab_size - 1, self._embedding_size],
          init=self._init_func)

      nil_word_slot = jnp.zeros([1, self._embedding_size])

    # This is "A" in the paper.
    if self._apply_embeddings:
      stories_biases = jnp.concatenate([stories_biases, nil_word_slot], axis=0)
      memory_embeddings = jnp.take(
          stories_biases, stories.reshape([-1]).astype(jnp.int32),
          axis=0).reshape(list(stories.shape) + [self._embedding_size])
      memory_embeddings = jnp.pad(
          memory_embeddings,
          ((0, 0), (0, self._memory_size - jnp.shape(memory_embeddings)[1]),
           (0, 0), (0, 0)))
      memory = jnp.sum(memory_embeddings * self._encodings, 2) + memory_biases
    else:
      memory = stories

    # This is "B" in the paper. Also, when there are no queries (only
    # sentences), then there these lines are substituted by
    # query_embeddings = 0.1.
    if self._apply_embeddings:
      query_biases = jnp.concatenate([query_biases, nil_word_slot], axis=0)
      query_embeddings = jnp.take(
          query_biases, queries.reshape([-1]).astype(jnp.int32),
          axis=0).reshape(list(queries.shape) + [self._embedding_size])
      # This is "u" in the paper.
      query_input_embedding = jnp.sum(query_embeddings * self._encodings, 1)
    else:
      query_input_embedding = queries

    # This is "C" in the paper.
    if self._apply_embeddings:
      output_biases = jnp.concatenate([output_biases, nil_word_slot], axis=0)
      output_embeddings = jnp.take(
          output_biases, stories.reshape([-1]).astype(jnp.int32),
          axis=0).reshape(list(stories.shape) + [self._embedding_size])
      output_embeddings = jnp.pad(
          output_embeddings,
          ((0, 0), (0, self._memory_size - jnp.shape(output_embeddings)[1]),
           (0, 0), (0, 0)))
      output = jnp.sum(output_embeddings * self._encodings, 2)
    else:
      output = stories

    intermediate_linear = hk.Linear(self._embedding_size, with_bias=False)

    # Output_linear is "H".
    output_linear = hk.Linear(self._linear_output_size, with_bias=False)

    for hop_number in range(self._num_hops):
      query_input_embedding_transposed = jnp.transpose(
          jnp.expand_dims(query_input_embedding, -1), [0, 2, 1])

      # Calculate probabilities.
      probs = jax.nn.softmax(
          jnp.sum(memory * query_input_embedding_transposed, 2))

      # Calculate output of the layer by multiplying by C.
      transposed_probs = jnp.transpose(jnp.expand_dims(probs, -1), [0, 2, 1])
      transposed_output_embeddings = jnp.transpose(output, [0, 2, 1])

      # This is "o" in the paper.
      layer_output = jnp.sum(transposed_output_embeddings * transposed_probs, 2)

      # Finally the answer
      if hop_number == self._num_hops - 1:
        # Please note that in the TF version we apply the final linear layer
        # in all hops and this results in shape mismatches.
        output_layer = output_linear(query_input_embedding + layer_output)
      else:
        output_layer = intermediate_linear(query_input_embedding + layer_output)

      query_input_embedding = output_layer
      if self._nonlin:
        output_layer = self._nonlin(output_layer)

    # This linear here is "W".
    ret = hk.Linear(self._vocab_size, with_bias=False)(output_layer)

    if self._use_ln:
      ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ret = ln(ret)

    return ret


class MemNetFull(MemNetMasked):
  """Memory Networks with full adjacency matrix."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


ProcessorFactory = Callable[[int], Processor]


def get_processor_factory(kind: str,
                          use_ln: bool,
                          nb_triplet_fts: int,
                          nb_heads: Optional[int] = None) -> ProcessorFactory:
  """Returns a processor factory.

  Args:
    kind: One of the available types of processor.
    use_ln: Whether the processor passes the output through a layernorm layer.
    nb_triplet_fts: How many triplet features to compute.
    nb_heads: Number of attention heads for GAT processors.
  Returns:
    A callable that takes an `out_size` parameter (equal to the hidden
    dimension of the network) and returns a processor instance.
  """
  def _factory(
    out_size: int,
    num_layers: int = 3,
    attention_dropout: float = 0.0,
    activation: str = 'relu',
    norm_first: bool = False,
  ):
    if kind == 'deepsets':
      processor = DeepSets(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=0
      )
    elif kind == 'gat':
      processor = GAT(
          out_size=out_size,
          nb_heads=nb_heads,
          use_ln=use_ln,
      )
    elif kind == 'edge_t':
      processor = ET_Processor(
          out_size=out_size,
          nb_heads=nb_heads,
          use_ln=use_ln,
          num_layers=num_layers,
          activation=activation,
          attention_dropout=attention_dropout,
          norm_first=norm_first,
      )
    elif kind == 'gat_full':
      processor = GATFull(
          out_size=out_size,
          nb_heads=nb_heads,
          use_ln=use_ln
      )
    elif kind == 'gatv2':
      processor = GATv2(
          out_size=out_size,
          nb_heads=nb_heads,
          use_ln=use_ln
      )
    elif kind == 'gatv2_full':
      processor = GATv2Full(
          out_size=out_size,
          nb_heads=nb_heads,
          use_ln=use_ln
      )
    elif kind == 'memnet_full':
      processor = MemNetFull(
          vocab_size=out_size,
          sentence_size=out_size,
          linear_output_size=out_size,
      )
    elif kind == 'memnet_masked':
      processor = MemNetMasked(
          vocab_size=out_size,
          sentence_size=out_size,
          linear_output_size=out_size,
      )
    elif kind == 'mpnn':
      processor = MPNN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=0,
      )
    elif kind == 'pgn':
      processor = PGN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=0,
      )
    elif kind == 'pgn_mask':
      processor = PGNMask(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=0,
      )
    elif kind == 'triplet_mpnn':
      processor = MPNN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
      )
    elif kind == 'triplet_pgn':
      processor = PGN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
      )
    elif kind == 'triplet_pgn_mask':
      processor = PGNMask(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
      )
    elif kind == 'gpgn':
      processor = PGN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=nb_triplet_fts,
          gated=True,
      )
    elif kind == 'gpgn_mask':
      processor = PGNMask(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=nb_triplet_fts,
          gated=True,
      )
    elif kind == 'gmpnn':
      processor = MPNN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=nb_triplet_fts,
          gated=True,
      )
    elif kind == 'triplet_gpgn':
      processor = PGN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
          gated=True,
      )
    elif kind == 'triplet_gpgn_mask':
      processor = PGNMask(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
          gated=True,
      )
    elif kind == 'triplet_gmpnn':
      processor = MPNN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
          gated=True,
      )
    else:
      raise ValueError('Unexpected processor kind ' + kind)

    return processor

  return _factory


def _position_encoding(sentence_size: int, embedding_size: int) -> np.ndarray:
  """Position Encoding described in section 4.1 [1]."""
  encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
  ls = sentence_size + 1
  le = embedding_size + 1
  for i in range(1, le):
    for j in range(1, ls):
      encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
  encoding = 1 + 4 * encoding / embedding_size / sentence_size
  return np.transpose(encoding)
