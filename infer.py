import argparse
from pathlib import Path

import librosa
import soundfile as sf
import torch
from tqdm import tqdm

from models.model_Sphere_VAE import build_model
from utils import utils


AUDIO_EXTENSIONS = {".flac", ".m4a", ".mp3", ".ogg", ".wav"}


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct audio with Sphere_VAE")
    parser.add_argument("--checkpoint", required=True, help="Generator checkpoint")
    parser.add_argument("--input", required=True, help="Audio file or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--config",
        default="configs/config_Sphere_VAE.yaml",
        help="Model configuration",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cuda:0 or cpu",
    )
    return parser.parse_args()


def find_audio(path):
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {path}")
    return sorted(
        item for item in path.rglob("*") if item.suffix.lower() in AUDIO_EXTENSIONS
    )


def load_model(config_path, checkpoint_path, device):
    hps = utils.get_hparams_from_file(config_path)
    model = build_model(
        d_model=hps.model.d_model,
        seanet_kwargs=hps.model.seanet,
        transformer_kwargs=hps.model.transformer,
        latent_dimension=hps.model.latent_dimension,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, hps.model.sample_rate


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, sample_rate = load_model(args.config, args.checkpoint, device)
    audio_files = find_audio(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_files:
        raise RuntimeError(f"No supported audio files found under {args.input}")

    with torch.inference_mode():
        for audio_path in tqdm(audio_files, desc="Reconstructing"):
            wav, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
            wav_tensor = torch.from_numpy(wav).view(1, 1, -1).to(device)
            latent, _ = model.encode(wav_tensor)
            reconstructed = model.decode(latent).squeeze().cpu().numpy()
            output_path = output_dir / f"{audio_path.stem}.wav"
            sf.write(output_path, reconstructed[: len(wav)], sample_rate)

    print(f"Wrote {len(audio_files)} file(s) to {output_dir}")


if __name__ == "__main__":
    main()
