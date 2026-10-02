# DELL

Spiking neural networks for event-camera vision, comparing end-to-end backpropagation with
block-local learning rules on the same architecture. One codebase covers three learning **modes**,
two **tasks** and two **datasets**:

- **Modes**
  - `e2e`: standard end-to-end backpropagation, one optimizer for the whole network.
  - `local`: block-local learning, with one detached loss and optimizer per block and a learned
    local readout head.
  - `decolle`: like `local`, but each block's readout is a fixed random projection that is never
    trained ([Kaiser et al., 2020](https://doi.org/10.3389/fnins.2020.00424)).
- **Tasks**: `optical_flow`, `semantic` (segmentation).
- **Datasets**: DSEC (optical flow and semantic); M3ED (semantic, with the M3ED-Semantic labels used by
  Kong et al. (EventFly, CVPR 2025), InternImage pseudo-labels warped into the event camera).

| Task | Datasets |
| --- | --- |
| `optical_flow` | `dsec` |
| `semantic` | `dsec`, `m3ed` |

The network is a fully spiking U-Net of leaky integrate-and-fire neurons, built on
[spikingjelly](https://github.com/fangwei123456/spikingjelly).

## Setup

Python 3.13 or newer.

```bash
pip install -r requirements.txt
pip install -e .
```

`dell/utils/neuron.py` uses spikingjelly's `cupy` backend, which needs a CUDA GPU and `cupy`
(`pip install cupy-cudaXXX` for your CUDA version). To run on CPU, change `'backend': 'cupy'` to
`'backend': 'torch'` there.

MLflow tracking is optional. To log to a remote server, copy `.env.example` to `.env` and fill in
`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`. Without a `.env`,
runs are tracked on the local filesystem.

## Data

Each dataset is downloaded on first use (with `tonic`, or from an AWS mirror for M3ED) into
`<data-path>/<Dataset>/`. Set `--data-path` to choose the location (default `~/data`).

Splits, all fixed in `dell/datasets/`:

- **DSEC optical flow**: 13 training and 5 validation recordings, the split of Cuadrado et al. and
  SDformerFlow (`dell/datasets/dsec.py`). The benchmark's test split is only used for submissions.
- **DSEC-Semantic**: the official train/test split.
- **M3ED**: `car_urban_day_city_hall`, `car_urban_day_penno_small_loop`, `car_urban_day_rittenhouse` and
  `car_urban_day_ucity_small_loop` for training; `car_urban_day_horse` and `car_urban_day_penno_big_loop`
  for validation (`dell/datasets/m3ed.py`). Split by recording, this is about 80/20 by sample count.

## Usage

Train a model:

```bash
python run.py --mode e2e     --task optical_flow --dataset dsec
python run.py --mode local   --task optical_flow --dataset dsec --batch-size 8 --depth 4
python run.py --mode decolle --task semantic     --dataset dsec
```

`python run.py --help` lists every option, and [docs/options.md](docs/options.md) explains them in
detail. The best checkpoint is uploaded to MLflow when training ends (`--save-checkpoint`).

Evaluate a trained checkpoint on the validation set:

```bash
python infer.py --mode e2e --task optical_flow --dataset dsec --checkpoint path/to/model.ckpt
python infer.py --mode e2e --task optical_flow --dataset dsec --mlflow-run-id <run_id>
```

Checkpoints do not store the model configuration, so every model and data flag used for training
(`--depth`, `--base-channels`, `--use-separable-activation`, ...) must be repeated for `infer.py` and the scripts
below. A mismatch only shows up as a shape error when the weights are loaded.

## Reproducing the paper

The defaults are the paper's settings: depth 4, 32 base channels, T=16 steps over a 100 ms window,
one sample per ground-truth frame (`--target-interval-s 0.1`), AdamW with a cosine schedule,
lr 5e-4, batch size 32, 100 epochs, seed 42, gradient clipping at 1.0. Deep supervision (the
multi-scale output average) is on for optical flow and off for semantic segmentation.

| Paper result | Command |
| --- | --- |
| Tables I and II, optical flow | `python run.py --mode {e2e,local,decolle} --task optical_flow --dataset dsec --checkpoint-monitor val/epe` |
| Table III, depth ablation | the same with `--mode {e2e,local} --depth {3,4,5}` |
| Table IV, DSEC-Semantic | `python run.py --mode {e2e,local,decolle} --task semantic --dataset dsec --checkpoint-monitor val/miou --checkpoint-mode max` |
| Table V, M3ED | `python run.py --mode {e2e,local,decolle} --task semantic --dataset m3ed --checkpoint-monitor val_car/miou --checkpoint-mode max` |
| Table VI, energy | `python scripts/energy_report.py` |

`local` is DELL and `decolle` is s-DECOLLE in the paper. Table I is the DSEC test benchmark: run
`scripts/download_dsec_test.py`, then `scripts/prepare_dsec_submission.py` on the trained checkpoint,
and upload the result to the [DSEC evaluation server](https://dsec.ifi.uzh.ch/).

The trained checkpoints are not released; train them with the commands above.

## Project layout

```
run.py                 Training entry point
infer.py               Evaluation of a checkpoint
dell/
  cli.py               Command-line options, and construction of the model and datamodule
  registry.py          Tables mapping --mode/--task/--dataset to classes
  blocks/              Encoder, decoder and bottleneck blocks, task heads, local-learning variants
  models/              U-Net trunk built from the blocks
  modules/             One LightningModule per mode and task: forward pass, losses, optimizers
  datasets/            LightningDataModules, raw datasets and augmentations
  utils/               Losses and metrics, MLflow helpers, logging mixins (firing rate, memory, video)
scripts/               Tools around run.py and infer.py, see below
docs/options.md        Detailed description of the command-line options
```

To add a model or a dataset, register it in `dell/registry.py`. `run.py` does not change.

## Scripts

All scripts are run from the repository root, for example `python scripts/download_model.py --help`.

| Script | Purpose |
| --- | --- |
| `download_model.py` | Download a checkpoint that `run.py` uploaded to your MLflow server. |
| `download_dsec_test.py` | Download the held-out test split of the DSEC optical-flow benchmark. |
| `prepare_dsec_submission.py` | Run a checkpoint on that test split and write the leaderboard submission. |
| `check_dsec_submission.py` | Validate a submission locally. This is a vendored copy of DSEC's own checker. |
| `check_predict.py` | Check that the label-free `predict()` used for submissions reproduces the validation metrics. |
| `energy_report.py` | Theoretical energy per inference (Loihi 1 and Eyeriss v1 figures) of this network against E-RAFT and Cuadrado et al. |
| `generate_validation_videos.py` | Render events, ground truth and predictions as videos over a validation set. |
| `generate_panel_columns.py` | Render individual panel columns as images, by running inference. |
| `extract_panel_columns.py` | Crop the columns out of a panel video logged during training. |

## License

MIT, see `LICENSE`.

## Citing

If this code is useful in your work, please cite the accompanying paper:

```bibtex
@article{vaillant2026dell,
  title   = {Toward On-Chip Training of Spiking Neural Networks for Dense Event-Based Vision},
  author  = {Vaillant, Maxime and Carlier, Axel and Ng, Lai Xing and Hurter, Christophe and Cottereau, Benoit R.},
  journal = {arXiv preprint arXiv:2609.32405},
  year    = {2026}
}
```
