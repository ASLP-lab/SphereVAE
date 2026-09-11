# Sphere_VAE

Training and inference code for Sphere_VAE, an audio variational
autoencoder with a Power-spherical posterior. This repository contains only
the Sphere_VAE model and its required components. Model weights and datasets are not included.

## Model

The model uses a SEANet encoder, a causal Transformer, a 64-dimensional
continuous spherical latent, a causal Transformer decoder, and a SEANet decoder.
The encoder predicts a direction and concentration for the Power-spherical
posterior. Samples are scaled to a sphere of radius `sqrt(64) = 8`.

- Audio: 24 kHz, mono.
- Encoder stride: `8 * 6 * 5 * 4 = 960` samples, giving 25 latent frames/s.
- `encode(audio)` returns a sampled latent and the mean KL loss.
- `encode_mu(audio)` returns the deterministic scaled direction.
- `decode(latent)` reconstructs the waveform.

The latent frame rate is determined by the audio sample rate divided by the
SEANet encoder stride; there is no separate `frame_rate` configuration.
The latent dimension is configured by `model.latent_dimension` (default: 64).

## Installation

Use Python 3.10 or newer. Install a PyTorch and torchaudio pair appropriate for
your machine, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

GPU training uses Accelerate. Configure it for your machine before launching:

```bash
accelerate config
```

## Training data

Place uncompressed TAR shards in `data/shards`, or update
`data.train_shards_dir` in `configs/config_Sphere_VAE.yaml`.
Each sample must contain matching entries:

```text
sample.mp3
sample.json
```

The JSON object may contain `id` and `text`. The loader caches TAR offsets in
`index_cache.pkl`; remove that cache after moving or changing the shards.
Audio is resampled to mono 24 kHz and fitted to 12-second training segments
by cropping, padding, or repeating short samples.

## Training

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch train.py -c configs/config_Sphere_VAE.yaml
# Multiple GPUs; configure the corresponding process count with Accelerate:
CUDA_VISIBLE_DEVICES=0,1 accelerate launch train.py -c configs/config_Sphere_VAE.yaml
```

Override the configuration with `CONFIG=path/to/config.yaml`. Checkpoints and
TensorBoard logs are written to `train.save_dir`. Existing generator and
discriminator checkpoints are loaded automatically when available.

To initialize fine-tuning from a generator-only checkpoint, place it in the
training directory using the `G_*.pth` naming convention:

```bash
mkdir -p exps/Sphere_VAE
cp model_weights/G_500000_model_only.pth exps/Sphere_VAE/G_500000.pth
```

The training script first attempts to restore the complete generator and
discriminator training state. If that is unavailable, it loads generator and
discriminator weights only; a missing discriminator checkpoint is allowed and
starts a fresh discriminator with `global_step=0`.

The generator objective combines adversarial loss, feature matching (weight
1.5), multi-scale Mel reconstruction (weight 15), and KL regularization (weight
0.01). A multi-scale STFT discriminator provides the adversarial objective.

### Existing training behavior

This export preserves the Sphere_VAE training algorithm. In particular:

- The loop updates optimizers every batch; the gradient accumulation setting
  is not implemented as accumulation across batches.
- `training_steps` controls the learning-rate scheduler, not the loop stop
  condition. The loop is primarily bounded by `epochs`.
- Optimizer betas and epsilon are specified in the training code.
- `eval_interval` saves checkpoints; there is no held-out validation loop.

## Inference

```bash
python infer.py \
  --checkpoint exps/Sphere_VAE/G_400000.pth \
  --input data/test \
  --output output/reconstructed \
  --device cuda:0
```

`--input` accepts an audio file or directory. Supported extensions are WAV,
FLAC, MP3, OGG, and M4A. Use `--config` to select another configuration.
Inference uses posterior sampling, so repeated runs can produce different
reconstructions. Output is trimmed to the input length.

## Layout

```text
models/model_Sphere_VAE.py  Sphere_VAE model and builder
modules/                SEANet, Transformer, spherical distribution, utilities
train.py                  Training entrypoint
infer.py                  Audio reconstruction entrypoint
dataset.py              TAR shard dataset
configs/                Sphere_VAE configuration
discriminators/         STFT discriminator
losses/                 Spectral and adversarial losses
utils/                  Configuration, checkpoints, and compilation utilities
```

## Acknowledgments

Parts of the code are adapted from Kyutai Mimi and Meta AudioCraft/EnCodec,
as indicated by the original notices retained in the source files. This
export does not assign a new license to that third-party code. The source
checkout did not include the root license files referenced by those notices;
applicable upstream license texts still need to be supplied before release.
