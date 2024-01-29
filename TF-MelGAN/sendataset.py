import os
import math
import random
import traceback
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.utils.data
import torch.nn.functional as F

from librosa.core import load
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    data, sample_rate = load(full_path, sr=8000)
    return data, sample_rate


def load_sensor(full_path):
    data = np.load(full_path)
    sr = 8000
    return data, sr


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_denormalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sample_rate, hop_size, win_size, fmin, fmax, center=False):
    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)
    return spec


class SenAudioDataset(torch.utils.data.Dataset):
    def __init__(self, training_audio_files, traingning_sensor_files, audio_seg_length, sensor_seg_length, sample_rate,
                 n_fft, num_mels, hop_size, win_size, fmin, fmax,
                 split=True, shuffle=False, n_cache_reuse=1, device=None, fmax_loss=None, fine_tuning=False, augment=True, test=False):
        self.sample_rate = sample_rate
        self.audio_seg_length = audio_seg_length
        self.sensor_seg_length = sensor_seg_length

        self.audio_files = self.files_to_list(training_audio_files)
        self.sensor_files = self.files_to_list(traingning_sensor_files)

        self.fmin = fmin
        self.fmax = fmax
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size

        self.test = test
        self.split = split
        self.device = device
        self.shuffle = shuffle
        self.augment = augment
        self.fmax_loss = fmax_loss
        self.fine_tuning = fine_tuning
        self.n_cache_reuse = n_cache_reuse

    def __getitem__(self, index):
        sensor_filepath = self.sensor_files[index]

        sensor = self.load_sensor_from_csv_to_torch(sensor_filepath, col="data", header=0)

        audio_filepath = self.audio_files[index]
        audio, sample_rate = self.load_audio_to_torch(audio_filepath)

        if sensor.size(0) >= self.sensor_seg_length:
            max_start = min(sensor.size(0), audio.size(0)) - self.sensor_seg_length
            start = random.randint(0, max_start)
            audio = audio[start: start + self.audio_seg_length]
            sensor = sensor[start: start + self.audio_seg_length]
        else:
            sensor = F.pad(sensor, (0, self.sensor_seg_length - sensor.size(0))).data
            audio = F.pad(audio, (0, self.audio_seg_length - audio.size(0))).data

        if audio.size(0) != self.audio_seg_length:
            audio = F.pad(audio, (0, self.audio_seg_length - audio.size(0))).data

        sensor = sensor.unsqueeze(0)
        audio = audio.unsqueeze(0)

        sensor_mel = mel_spectrogram(sensor, self.n_fft, self.num_mels,
                                     self.sample_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                     center=False)
        audio_mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                    self.sample_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                    center=False)

        return (sensor, sensor_mel.squeeze(), audio, audio_mel.squeeze())

    def __len__(self):
        return len(self.sensor_files)

    def files_to_list(self, filename):
        with open(filename, encoding="utf-8") as f:
            files = f.readlines()

        files = [f.rstrip() for f in files]
        return files

    def load_audio_to_torch(self, audio_full_path):
        data, sampling_rate = load(audio_full_path, sr=self.sample_rate)
        data = 0.95 * normalize(data)
        return torch.from_numpy(data).float(), sampling_rate

    def load_sensor_from_npy_to_torch(self, sensor_full_path):

        data = np.load(sensor_full_path)
        return torch.from_numpy(data).float()

    def load_sensor_from_csv_to_torch(self, sensor_full_path, col="data", header=0):
        data = pd.read_csv(sensor_full_path, header=header)[col].to_numpy()
        return torch.from_numpy(data).float()
