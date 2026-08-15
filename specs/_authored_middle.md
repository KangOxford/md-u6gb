## Repository Structure

The `lob/` package is organized **by pipeline stage** — each stage of the model
lifecycle lives in its own subpackage. Start here.

```
lob/                          LOB model source (S5), organized by pipeline stage
├── encode/                   Message tokenization (LOBSTER message <-> token ids)
│   ├── encoding.py               Canonical tokenizer: Message_Tokenizer, Vocab (26-token mode)
│   ├── encoding_1tok.py          Per-field vocab sizes + local<->global token mapping
│   ├── encoding_22tok.py         Legacy 22-token format (selectable in run_inference.py)
│   ├── encoding_26tok.py         26-token format (selectable in run_inference.py)
│   └── encoding_23tok.py …       Archived variants (incl. encoding_24tok.py) — not imported
├── preprocess/               Raw LOBSTER data -> arrays -> batched tensors
│   ├── preproc.py                .csv -> .npy preprocessing (run as: python -m lob.preprocess.preproc)
│   ├── lobster_dataloader.py     LOBSTER_Dataset (PyTorch), file caching, masking
│   └── dataloading.py            Dataset factory + DistributedSampler
├── model/                    Network definition
│   └── lob_seq_model.py          PaddedLobPredModel: stacked S5 layers (message + book fusion)
├── train/                    Training loop + distributed infrastructure
│   ├── train.py                  train(): epoch loop, validation, checkpointing, W&B
│   ├── train_helpers.py          JIT train/eval steps, cross-entropy loss, LR schedule, shard_map
│   ├── init_train.py             Model/TrainState init, Orbax checkpoint load/save
│   ├── sharding_utils.py         JAX mesh (1D flat / 2D hierarchical), data & param shardings
│   └── sweep.py                  W&B hyperparameter sweep
├── infer/                    Autoregressive generation
│   ├── inference.py              Generation with error correction
│   └── inference_no_errcorr.py   Generation without error correction
└── evaluate/                 Metrics
    ├── evaluation.py             Evaluation logic
    └── validation_helpers.py     Validation utilities

s5/                           S5 state-space core (and Mamba3 / GDN / FLA variants)
m3_kernels/                   CUDA kernels for the Mamba3 state scan
bin/                          Experiment launch scripts
run_train.py                  Training entry point      -> lob.train.train.train()
run_eval.py                   Evaluation entry point
run_inference.py              Inference entry point
generate_data.py              Synthetic message generation from a trained model
train_full_autoreg.batch      SLURM launch script (see "Model & Training Reference")
```

### Subpackage responsibilities

| Subpackage | Responsibility | Key public symbols |
|:-----------|:---------------|:-------------------|
| `lob.encode` | Tokenize messages <-> ids | `Message_Tokenizer`, `Vocab`, `encode_msgs` |
| `lob.preprocess` | Raw data, datasets, loaders | `LOBSTER_Dataset`, `create_lobster_prediction_dataset`, `preproc` (CLI) |
| `lob.model` | S5 sequence model | `PaddedLobPredModel` (+ batched variants) |
| `lob.train` | Optimisation + multi-GPU | `train`, `create_train_state`, `init_train_state`, `initialize_mesh` |
| `lob.infer` | Autoregressive generation | `inference`, `inference_no_errcorr` |
| `lob.evaluate` | Metrics + validation | `evaluation`, `validation_helpers` |

### Import paths

Imports follow the subpackage layout:

```python
from lob.encode.encoding import Message_Tokenizer, Vocab
from lob.preprocess.dataloading import create_lobster_prediction_dataset
from lob.model.lob_seq_model import PaddedLobPredModel
from lob.train.train import train                  # the train() function
from lob.train.train_helpers import create_train_state
from lob.infer import inference_no_errcorr
from lob.evaluate import evaluation
```

## Data

The data used is NASDAQ LOB data from [LOBSTER](https://lobsterdata.com/index.php).
After downloading and unpacking, the raw files must be pre-processed into the
arrays the model consumes (see Quickstart step 2).

## Quickstart

```bash
# 1. Install (verified package versions are listed in the manifest below)
pip install -r requirements.txt

# 2. Preprocess LOBSTER data.
#    Run as a MODULE (-m): preproc.py now lives inside the lob.preprocess package,
#    so `python lob/preprocess/preproc.py` would not find `import lob`.
python -m lob.preprocess.preproc \
    --data_dir /path/to/LOBS5/data/GOOG/ \
    --save_dir /path/to/LOBS5/data/GOOG/ \
    --n_tick_range 500 --use_raw_book_repr

# 3. Train: single process, or launch the multi-node SLURM job
python run_train.py                  # see `python run_train.py --help` for args
sbatch train_full_autoreg.batch

# 4. Evaluate / generate
python run_eval.py
python run_inference.py
```

> 📖 **The "Model & Training Reference" section below documents the data format,
> model architecture, loss/optimization (with rendered math), distributed
> training, and reported metrics. The SLURM script `train_full_autoreg.batch`
> points back here.**
