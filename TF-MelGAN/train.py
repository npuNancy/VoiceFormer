import os
import sys
import time
import json
import shutil
import pathlib
import argparse
import warnings
import itertools
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group
from torch.utils.data import DistributedSampler, DataLoader, dataloader
from torch.utils.tensorboard import SummaryWriter

from utils import save_sample
from models import feature_loss, generator_loss, discriminator_loss
from models import generator_loss_for_wd, discriminator_loss_for_wd
from models import WaveletDiscriminator
from models import Generator, MultiScaleDiscriminator
from sendataset import SenAudioDataset, mel_spectrogram


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))


torch.backends.cudnn.benchmark = True
warnings.simplefilter(action='ignore', category=FutureWarning)


def train(rank, a, h):
    root = Path(a.checkpoint_path)
    root.mkdir(exist_ok=True, parents=True)

    writer = SummaryWriter(str(a.checkpoint_path))
    print("writter success")

    generator = Generator(h).cuda()
    wd = WaveletDiscriminator().cuda()
    msd = MultiScaleDiscriminator().cuda()
    print("model loaded success")

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), wd.parameters()), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    print("create optimizer success")

    load_root = Path(a.load_path) if a.load_path else None
    if load_root and load_root.exists():

        generator.load_state_dict(torch.load(load_root / "generator.pt"))
        msd.load_state_dict(torch.load(load_root / "msd.pt"))
        wd.load_state_dict(torch.load(load_root / "wd.pt"))

        optim_g.load_state_dict(torch.load(load_root / "optim_g.pt"))
        optim_d.load_state_dict(torch.load(load_root / "optim_d.pt"))

    print(
        f"-----{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}: load model success----")

    print(
        f"-----{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}: strat loading dataset----")

    train_set = SenAudioDataset(
        Path(a.path) / "train_wav_files.txt",
        Path(a.path) / "train_sen_files.txt",
        audio_seg_length=h.segment_size,
        sensor_seg_length=h.segment_size,
        sample_rate=h.sampling_rate,
        n_fft=h.n_fft,
        num_mels=h.num_mels,
        hop_size=h.hop_size,
        win_size=h.win_size,
        fmin=h.fmin,
        fmax=h.fmax
    )

    test_set = SenAudioDataset(
        Path(a.path) / "test_wav_files.txt",
        Path(a.path) / "test_sen_files.txt",
        audio_seg_length=h.segment_size,
        sensor_seg_length=h.segment_size,
        sample_rate=h.sampling_rate,
        n_fft=h.n_fft,
        num_mels=h.num_mels,
        hop_size=h.hop_size,
        win_size=h.win_size,
        fmin=h.fmin,
        fmax=h.fmax,
        test=True
    )
    print("data set loaded success")

    train_loader = DataLoader(train_set, batch_size=h.batch_size, num_workers=h.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1)

    print(
        f"-----{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}: dataLoader prepared ----")

    test_voc = []
    test_audio = []
    for i, (sen_wav, sen_mel, audio_wav, audio_mel) in enumerate(test_loader):
        sen_wav = sen_wav.cuda()
        sen_mel = sen_mel.cuda()
        audio_wav = audio_wav.cuda()
        audio_mel = audio_mel.cuda()

        test_voc.append(sen_mel)
        test_audio.append(audio_wav)

        audio_wav = audio_wav.squeeze().cpu()
        save_sample(root / (f"original_{i}.wav"),
                    sampling_rate=h.sampling_rate, audio=audio_wav)
        writer.add_audio(
            f"original/sample_{i}.wav", audio_wav, 0, sample_rate=h.sampling_rate)

        if i == a.n_test_samples - 1:
            break

    print("original audio save success")

    costs = []
    start_time = time.time()

    torch.backends.cudnn.benchmark = True

    print(
        f"-----{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}: Training start ----")

    best_mel_reconst = 1000000
    steps = 0
    index = 0

    generator.train()
    wd.train()
    msd.train()
    for epoch in range(1, a.training_epochs+1):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} Epoch number: {epoch}")
        for iterno, (sen_wav, sen_mel, audio_wav, audio_mel) in enumerate(train_loader):
            sen_wav = sen_wav.cuda()
            sen_mel = sen_mel.cuda()
            audio_wav = audio_wav.cuda()
            audio_mel = audio_mel.cuda()

            audio_wav_gen = generator(sen_mel)
            audio_mel_gen = mel_spectrogram(audio_wav_gen.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                            h.fmin, h.fmax_for_loss)

            optim_d.zero_grad()

            wd_real, _ = wd(audio_wav)
            wd_gen, _ = wd(audio_wav_gen.detach())
            wd_loss, wd_loss_real, wd_loss_gen = discriminator_loss_for_wd(wd_real, wd_gen)

            msd_real, msd_gen, _, _ = msd(audio_wav, audio_wav_gen.detach())
            msd_loss, msd_loss_real, msd_loss_gen = discriminator_loss(msd_real, msd_gen)

            discriminator_loss_all = msd_loss + wd_loss
            discriminator_loss_all.backward()
            optim_d.step()

            optim_g.zero_grad()

            mel_loss = F.l1_loss(audio_mel, audio_mel_gen) * 45

            wd_real, wd_fmap_real = wd(audio_wav)
            wd_gen, wd_fmap_gen = wd(audio_wav_gen.detach())
            msd_real, msd_gen, msd_fmap_real, msd_fmap_gen = msd(audio_wav, audio_wav_gen.detach())

            wd_feature_loss = feature_loss(wd_fmap_real, wd_fmap_gen)
            msd_feature_loss = feature_loss(msd_fmap_real, msd_fmap_gen)

            gen_loss_wd, gen_loss_map_wd = generator_loss_for_wd(wd_gen)
            gen_loss_msd, gen_loss_map_msd = generator_loss(msd_gen)

            generator_loss_all = gen_loss_wd + gen_loss_msd + msd_feature_loss + wd_feature_loss + mel_loss
            generator_loss_all.backward()
            optim_g.step()

            costs.append([generator_loss_all.item(), discriminator_loss_all.item(), wd_loss.item(), msd_loss.item(), mel_loss.item()])

            writer.add_scalar("loss/generator", costs[-1][0], steps)
            writer.add_scalar("loss/discriminator", costs[-1][1], steps)
            writer.add_scalar("loss/Wavelet_disc_loss", costs[-1][2], steps)
            writer.add_scalar("loss/multi_scale_disc_loss", costs[-1][3], steps)
            writer.add_scalar("loss/mel_loss", costs[-1][4], steps)

            steps += 1

            if steps % a.checkpoint_interval == 0:
                save_time = time.time()

                with torch.no_grad():
                    for i, (sen_mel, _) in enumerate(zip(test_voc, test_audio)):
                        audio_wav_pred = generator(sen_mel)
                        audio_wav_pred = audio_wav_pred.squeeze().cpu()
                        save_sample(root / ("generated_%d.wav" % i), 8000, audio_wav_pred)
                        writer.add_audio(
                            "generated/sample_%d.wav" % i,
                            audio_wav_pred,
                            epoch,
                            sample_rate=h.sampling_rate
                        )

                torch.save(generator.state_dict, root / "generator.pt")
                torch.save(wd.state_dict(), root / "wd.pt")
                torch.save(msd.state_dict(), root / "msd.pt")

                torch.save(optim_g.state_dict(), root / "optim_g.pt")
                torch.save(optim_d.state_dict(), root / "optim_d.pt")

                if np.asarray(costs).mean(0)[-1] < best_mel_reconst:
                    best_mel_reconst = np.asarray(costs).mean(0)[-1]
                    torch.save(generator.state_dict(), root / "best_generator.pt")
                    torch.save(wd.state_dict(), root / "best_wd.pt")
                    torch.save(msd.state_dict(), root / "best_msd.pt")

                print("Took %5.4fs to generate samples" % (time.time() - save_time))
                print("-" * 100)

            if steps % a.summary_interval == 0:

                print(
                    "Epoch {} | Iters {} / {} | ms/batch {:5.2f} | loss {}".format(
                        epoch,
                        iterno,
                        len(train_loader),
                        1000 * (time.time() - start_time) / a.summary_interval,
                        np.asarray(costs).mean(0),
                    )
                )
                costs = []
                start_time = time.time()


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--config', default='config_sen.json')
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--training_epochs', default=1000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--checkpoint_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    parser.add_argument("--n_test_samples", default=16)

    parser.add_argument("--load_path", default=None)
    parser.add_argument('--path', default=None, required=True)

    a = parser.parse_args()

    with open(a.config, 'r', encoding='utf-8') as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
    else:
        pass

    torch.cuda.set_device(0)

    train(0, a, h)


if __name__ == '__main__':
    main()
