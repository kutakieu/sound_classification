import torch
from torch.utils.data import Dataset
# from torchvision import transforms
import librosa
from librosa.effects import pitch_shift
from pathlib import Path
import numpy as np
import random

from src.sound_transforms import Random_slice, Mixup, Random_Noise

classes = ["Acoustic_guitar", "Applause", "Bark", "Bass_drum", "Burping_or_eructation", "Bus", "Cello", "Chime",
           "Clarinet", "Computer_keyboard", "Cough", "Cowbell", "Double_bass", "Drawer_open_or_close", "Electric_piano",
           "Fart", "Finger_snapping", "Fireworks", "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire", "Harmonica",
           "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow", "Microwave_oven", "Oboe", "Saxophone", "Scissors",
           "Shatter", "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone", "Trumpet", "Violin_or_fiddle",
           "Writing"]

class AudioDataset(Dataset):
    def __init__(self, data_dir, data_list_file, transform):
        self.n_fft = 1024
        self.sr = 32000
        self.mono = True
        self.log_spec = False
        self.n_mels = 128
        self.hop_length = 192
        self.fmax = None

        fin = open(data_list_file, "r")
        lines = fin.readlines()

        # tmp = pathlib.Path(data_dir)
        # self.files = list(tmp.glob('**/*.txt'))
        self.data = []
        self.labels = []
        for line in lines[1:]:
            filename, label = line.split(",")[:2]
            filename = filename.replace(".wav", ".npy")
            label_tmp = np.zeros((len(classes)), dtype=np.float32)
            label_tmp[classes.index(label)] = 1.0
            sig = np.load(data_dir / filename)[0]
            self.data.append(sig)
            self.labels.append(label_tmp)

            # """apply pitch_shift -3~3"""
            # for i in range(-3,3):
            #     spectrogram = self.preprocess(data_dir / filename, pitch_shift=i)
            #     if i != 0:
            #         sig = librosa.effects.pitch_shift(sig_orig, self.sr, n_steps=i)
            #     else:
            #         sig = sig_orig
            #     self.data.append(sig)
            #     self.labels.append(label_tmp)

        self.transform = transform
        self.random_noise = self.preprocess("./data/enviroment1.mp3")

        print(len(self.data))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        x2mix = self.data[random.randint(0,len(self.data)-1)]
        y2mix = self.labels[random.randint(0,len(self.data)-1)]
        # if self.transform:
        """Random_slice"""
        x, y, x2mix, y2mix = self.transform[0]([x, y, x2mix, y2mix])
        if len(self.transform) > 1:
            """Mixup"""
            x, y = self.transform[1]([x, y, x2mix, y2mix])
            """Random_Noise"""
            x, y = self.transform[2]([x, y, self.random_noise])
        # y = torch.LongTensor(np.argmax(y))
        # y = np.argmax(y)
        # print(x)
        return np.float32(x), y
        # return np.float32(x), np.argmax(y)


    def preprocess(self, file_path, **kwargs):
        if self.mono:
            sig, sr = librosa.load(file_path, sr=self.sr, mono=True)
            sig = sig[np.newaxis]
        else:
            sig, sr = librosa.load(file_path, sr=self.sr, mono=False)
            # sig, sf_sr = sf.read(file_path)
            # sig = np.transpose(sig, (1, 0))
            # sig = np.asarray([librosa.resample(s, sf_sr, sr) for s in sig])
        # sig = librosa.effects.pitch_shift(sig, sr, n_steps=pitch_shift)

        for y in sig:

            # compute stft
            stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=None, window='hann', center=True,
                                pad_mode='reflect')

            # keep only amplitures
            stft = np.abs(stft)

            # spectrogram weighting
            if self.log_spec:
                stft = np.log10(stft + 1)
            else:
                freqs = librosa.core.fft_frequencies(sr=sr, n_fft=self.n_fft)
                stft = librosa.perceptual_weighting(stft**2, freqs, ref=1.0, amin=1e-10, top_db=99.0)

            # apply mel filterbank
            spectrogram = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=self.n_mels, fmax=self.fmax)

        # print(sig.shape)
        # print(spectrogram.shape)
        return spectrogram


if __name__ == '__main__':
    transform = transforms.Compose([Random_slice(), Mixup()])
    dataset = AudioDataset(Path("data/processed_data"), \
        "test_data_prep.txt", transform=[Random_slice(), Mixup(), Random_Noise()])

    print(dataset[0][0].shape)
    print(dataset[1][0].shape)
    print(dataset[2][0].shape)
