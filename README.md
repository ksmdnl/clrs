# Towards Principled Graph Transformers (CLRS-30)
Code to reproduce our results on CLRS-30.

## Install
We recommend to use the package manager [`conda`](https://docs.conda.io/en/latest/). Once installed run

```bash
conda create -n towards-principled-gts python=3.10

conda activate towards-principled-gts
```
Install all dependencies via:
```bash
pip install -e .
```
## Configuration
By default, logging with `wandb` is disabled. To enable it set `--wandb_project` in the command line.  Optionally, set `--wandb_entity` and `--wandb_name` to configure your entity and run name, respectively.

## Experiments
To benchmark the ET on e.g., Dijkstra, run

```bash
python -m clrs.examples.run --algorithms dijkstra
```

## CLRS-30 Benchmark
For more detailed explanation about the benchmark refer to [clrs repository](https://github.com/google-deepmind/clrs) and the [CLRS Paper](https://arxiv.org/abs/2205.15659).