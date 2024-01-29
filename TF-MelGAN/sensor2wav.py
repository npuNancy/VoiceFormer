

import os
import json
import yaml
import torch
import argparse
import numpy as np
import pandas as pd
import scipy.io.wavfile
from tqdm import tqdm
from pathlib import Path

from models import Generator
from sendataset import mel_spectrogram


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def load_model(mel2wav_path, device=get_default_device()):
    root = Path(mel2wav_path)
    with open(root / "args.yml", "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    netG = Generator(args.n_mel_channels, args.ngf, args.n_residual_layers).to(device)
    netG.load_state_dict(torch.load(root / "netG.pt", map_location=device))
    return netG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=Path, required=True)
    parser.add_argument("--test_path", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument('--config', default='config_sen.json')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    model = Generator(h).cuda()
    model.load_state_dict(torch.load(args.load_path / "best_generator.pt", map_location="cuda"))

    args.save_path.mkdir(exist_ok=True, parents=True)

    with open(args.test_path, "r") as f:
        test_files = f.readlines()

    for i, file_path in enumerate(test_files):
        file_path = file_path.strip().replace('\n', '')
        file_name = os.path.basename(file_path).strip().replace('.csv', '')
        sensor_data = pd.read_csv(file_path, header=0)["data"].to_numpy()
        sen = sensor_data

        sr = 8000
        amplitude = np.random.uniform(low=0.3, high=1.0)
        sen = sen * amplitude
        sen = sen.astype(np.float32)
        sen = torch.from_numpy(sen).float().unsqueeze(0)
        mel = mel_spectrogram(sen, n_fft=256, num_mels=80, sample_rate=8000, hop_size=64, win_size=256, fmin=0, fmax=4000)
        mel = mel.cuda()
        recons = model(mel).squeeze().cpu()
        recons = recons.detach().numpy()
        audio = (recons * 32768).astype("int16")
        save_path = os.path.join(str(args.save_path), f"{file_name}.wav")
        scipy.io.wavfile.write(save_path, sr, audio)
        print(f'{file_name}: generated ...')


if __name__ == "__main__":
    main()
