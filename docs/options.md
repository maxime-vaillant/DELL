# Command-line options

`python run.py --help` lists every option with a one-line description. This page adds the details.
`infer.py` and the scripts in `scripts/` accept the same model and data options, which must match the
training run: checkpoints do not store them.

Options in the **model**, **logging** and **data** groups are only passed on when the chosen
model or dataset has a matching constructor argument. Otherwise they are ignored and a notice is
printed (`[run] <Class> ignores: ...`). For example `--use-separable-activation` has no effect with `--mode decolle`.

## Learning modes

| `--mode` | Training |
| --- | --- |
| `e2e` | Standard backpropagation through the whole network with one optimizer. |
| `local` | Every block (each encoder block, the bottleneck, each decoder block and the trunk head) has its own head, loss and optimizer. Activations are detached between blocks, so no gradient crosses a block boundary. |
| `decolle` | Like `local`, but the heads of the encoder blocks, the bottleneck and the decoder blocks are fixed random readouts that are never trained. Gradients still flow through them into their block. The trunk head is trained. |

## Run

| Option | Notes |
| --- | --- |
| `--gradient-clip-val` | Gradient-norm clipping, 0 disables it. With `local` and `decolle`, each block's gradient is clipped on its own, which is looser than `e2e`'s clip over the whole network, so the same value is not equivalent across modes. |
| `--overfit-batches N` | Trains and validates on `N` batches only, and turns off video logging. |
| `--no-val` | Skips the sanity check and validation. Incompatible with `--save-checkpoint best`. |

## Checkpoints

With `--save-checkpoint last` or `best`, the weights are saved locally in `--checkpoint-dir` and uploaded
once to MLflow when training ends (`checkpoints/model.ckpt` in the run's artifacts). Download them with
`scripts/download_model.py`, or pass `--mlflow-run-id` to `infer.py`.

`--checkpoint-monitor` must be a metric logged by the module's validation step, for example `val/loss`,
`val/epe` or `val/miou`. It is not validated up front: an unknown name fails at the end of the first
validation epoch. Semantic segmentation on M3ED logs `val_car/...` instead of `val/...`.

## Model

| Option | Notes |
| --- | --- |
| `--use-separable-activation` | Adds a neuron between the depthwise and pointwise convolutions of every separable convolution (the deepest encoder stage, the first decoder stage and the bottleneck). `e2e` and `local` only. |
| `--use-deep-supervision` | On by default for optical flow, off for semantic segmentation, as in the paper. See below. |

### Deep supervision

- **Optical flow, `e2e`**: every decoder block gets a flow head. The loss is the mean of the trunk head's
  loss and every decoder loss, backpropagated once. The reported EPE, AE and NPE, and the logged panels,
  average all head predictions, each upsampled to the ground truth resolution.
- **Optical flow, `local` and `decolle`**: training is unchanged, as every block already has its own head
  and loss. Only the reported metrics, panels and `predict()` change: they average the decoder blocks'
  predictions with the trunk head's.
- **Semantic, `e2e`**: every decoder block gets a semantic head and the loss is the mean of all head
  losses. Metrics and panels always score the trunk head alone.
- **Semantic, `local` and `decolle`**: no effect.

## Image and video logging

Validation comparison panels and train/validation mp4 videos are logged to MLflow (ffmpeg is needed
for the videos). By default only the last epoch is logged.

| Option | Notes |
| --- | --- |
| `--log-video` | Master switch for panels and videos. |
| `--log-last-epoch-only` | Log only the final epoch. With `--no-log-last-epoch-only`, log every `--log-images-every-n-epochs` epochs, skipping epoch 0. |
| `--max-video-frames` | Frames are kept in memory until the video is encoded. This caps them, subsampled evenly over the epoch. Lower it if you run out of memory, for example with a small `--target-interval-s` on M3ED. |

## Data

| Option | Notes |
| --- | --- |
| `--data-path` | Dataset root, `~/data` by default. |
| `--target-type` | DSEC only: `optical_flow` or `semantic`. Defaults to `semantic` with `--task semantic`, otherwise `optical_flow`. |
| `--window-duration-s` | DSEC and M3ED: length of the event window ending at each target. |
| `--target-interval-s` | DSEC and M3ED: minimum time between two samples. The default, 0.1 s, keeps every DSEC flow frame (10 Hz). A larger value shrinks the dataset. |
| `--label-taxonomy` | DSEC and M3ED semantic. On M3ED, `cityscapes-11` merges the 19 InternImage classes into 11 (`utils.semantic.CITYSCAPES_19_TO_11`), as several classes have almost no examples. On DSEC it selects which of the dataset's own label folders to read. The default is `cityscapes-11` on both, as in the paper. |

`--event-view`, `--rectify` and `--download` are fixed by `run.py` to `left`, on and on.
