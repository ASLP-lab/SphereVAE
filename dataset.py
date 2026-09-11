import io
import json
import os
import pickle
import random
import tarfile

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


class Mp3TarOffsetDataset(Dataset):
    """Load paired MP3/JSON samples directly from uncompressed tar shards."""

    def __init__(
        self,
        base_path,
        sr=24000,
        segment_size=288000,
        cache_path=None,
        seed_value=42,
    ):
        self.base_path = os.path.abspath(os.path.expanduser(base_path))
        self.sr = sr
        self.segment_size = segment_size
        self.samples = []

        if cache_path is None:
            cache_path = os.path.join(self.base_path, "index_cache.pkl")

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as handle:
                self.samples = pickle.load(handle)
        else:
            self.samples = self._scan_shards()
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as handle:
                pickle.dump(self.samples, handle)

        random.Random(seed_value).shuffle(self.samples)
        print(f"Loaded {len(self.samples)} samples from {self.base_path}")

    def _scan_shards(self):
        samples = []
        for root, _, files in os.walk(self.base_path, followlinks=True):
            for filename in sorted(files):
                if not filename.endswith(".tar"):
                    continue
                tar_path = os.path.join(root, filename)
                with tarfile.open(tar_path, "r") as archive:
                    members = {member.name: member for member in archive.getmembers()}
                    for json_name, json_info in members.items():
                        if not json_name.endswith(".json"):
                            continue
                        mp3_info = members.get(json_name[:-5] + ".mp3")
                        if mp3_info is None:
                            continue
                        samples.append(
                            (
                                tar_path,
                                mp3_info.offset_data,
                                mp3_info.size,
                                json_info.offset_data,
                                json_info.size,
                            )
                        )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        tar_path, mp3_offset, mp3_size, json_offset, json_size = self.samples[index]
        try:
            with open(tar_path, "rb") as handle:
                handle.seek(json_offset)
                metadata = json.loads(handle.read(json_size).decode("utf-8"))
                handle.seek(mp3_offset)
                mp3_bytes = handle.read(mp3_size)

            wav, _ = librosa.load(io.BytesIO(mp3_bytes), sr=self.sr, mono=True)
            wav = self._fit_segment(wav)
            return {
                "wav": torch.from_numpy(wav.copy()).float(),
                "utt": metadata.get("id", "unknown"),
                "text": metadata.get("text", ""),
            }
        except Exception as error:
            print(f"Skipping sample {index} from {tar_path}: {error}")
            return None

    def _fit_segment(self, wav):
        wav_length = len(wav)
        if wav_length == 0:
            raise ValueError("decoded an empty audio sample")
        if wav_length > self.segment_size:
            start = np.random.randint(0, wav_length - self.segment_size + 1)
            return wav[start : start + self.segment_size]
        if wav_length >= int(0.8 * self.segment_size):
            return np.pad(wav, (0, self.segment_size - wav_length))

        repeats = int(np.ceil(self.segment_size / wav_length))
        repeated = np.tile(wav, repeats)
        start = np.random.randint(0, repeated.size - self.segment_size + 1)
        return repeated[start : start + self.segment_size]
