import os
import random

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_scp(scp_path):
    """Parse ``utt_id path`` or single-path SCP lines."""
    samples = []
    scp_dir = os.path.dirname(os.path.abspath(os.path.expanduser(scp_path)))
    with open(scp_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(None, 1)
            if len(parts) == 1:
                audio_path = parts[0]
                utt = os.path.splitext(os.path.basename(audio_path))[0]
            else:
                utt, audio_path = parts

            audio_path = os.path.expanduser(audio_path)
            if not os.path.isabs(audio_path):
                # Support both OmniCodec-style paths relative to the launch
                # directory and paths relative to the SCP file itself.
                if os.path.exists(audio_path):
                    audio_path = os.path.abspath(audio_path)
                else:
                    audio_path = os.path.join(scp_dir, audio_path)
            samples.append((utt, audio_path, line_number))
    return samples


class ScpAudioDataset(Dataset):
    """Load mono audio listed in an SCP file and return fixed-size segments."""

    def __init__(
        self,
        scp_path,
        sr=24000,
        segment_size=288000,
        seed_value=42,
    ):
        self.scp_path = os.path.abspath(os.path.expanduser(scp_path))
        self.sr = sr
        self.segment_size = segment_size
        self.seed_value = seed_value
        self.samples = _load_scp(self.scp_path)
        random.Random(seed_value).shuffle(self.samples)
        print(f"Loaded {len(self.samples)} samples from {self.scp_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        utt, audio_path, line_number = self.samples[index]
        try:
            wav, _ = librosa.load(audio_path, sr=self.sr, mono=True)
            if wav.size == 0:
                raise ValueError("decoded an empty audio sample")

            # Match the OmniCodec input convention.
            wav = librosa.util.normalize(wav) * 0.95
            wav = self._fit_segment(wav)
            return {
                "wav": torch.from_numpy(wav.copy()).float(),
                "utt": utt,
                "text": "",
            }
        except Exception as error:
            print(
                f"Skipping SCP line {line_number} ({utt}) from "
                f"{self.scp_path}: {error}"
            )
            return None

    def _fit_segment(self, wav):
        wav_length = len(wav)
        if wav_length > self.segment_size:
            start = np.random.randint(0, wav_length - self.segment_size + 1)
            return wav[start : start + self.segment_size]
        if wav_length >= int(0.8 * self.segment_size):
            return np.pad(wav, (0, self.segment_size - wav_length))

        repeats = int(np.ceil(self.segment_size / wav_length))
        repeated = np.tile(wav, repeats)
        start = np.random.randint(0, repeated.size - self.segment_size + 1)
        return repeated[start : start + self.segment_size]
