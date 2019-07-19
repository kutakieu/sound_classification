import torch
from torch.utils.data import Dataset
# from torchvision import transforms
import librosa
from librosa.effects import pitch_shift
from pathlib import Path
import numpy as np
import random

from src.sound_transforms import Random_slice, Mixup, Random_Noise
from src.feature.utils import *

classes = ["Acoustic_guitar", "Applause", "Bark", "Bass_drum", "Burping_or_eructation", "Bus", "Cello", "Chime",
           "Clarinet", "Computer_keyboard", "Cough", "Cowbell", "Double_bass", "Drawer_open_or_close", "Electric_piano",
           "Fart", "Finger_snapping", "Fireworks", "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire", "Harmonica",
           "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow", "Microwave_oven", "Oboe", "Saxophone", "Scissors",
           "Shatter", "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone", "Trumpet", "Violin_or_fiddle",
           "Writing"]

class AudioDataset(Dataset):
    def __init__(self, data_dir, data_list_file, transform, signal_ratio=16000):
        # self.n_fft = 1024
        self.sr = signal_ratio
        self.mono = True
        # self.log_spec = False
        # self.n_mels = 128
        # self.hop_length = 192
        # self.fmax = None

        self.transform = transform

        fin = open(data_list_file, "r")
        lines = fin.readlines()

        # tmp = pathlib.Path(data_dir)
        # self.files = list(tmp.glob('**/*.txt'))
        self.data = []
        self.labels = []
        self.filenames = []
        for line in lines[1:]:
            filename, label = line.split(",")[:2]
            filename = filename.replace(".wav", ".npy")
            self.filenames.append(data_dir / filename)
            # sig = np.load(data_dir / filename)
            # self.data.append(sig)

            # if label is onehot:
                # label_tmp = np.zeros((len(classes)), dtype=np.float32)
                # label_tmp[classes.index(label)] = 1.0
                # self.labels.append(label_tmp)
            self.labels.append(np.array(classes.index(label)))


            # """apply pitch_shift -3~3"""
            # for i in range(-3,3):
            #     spectrogram = self.preprocess(data_dir / filename, pitch_shift=i)
            #     if i != 0:
            #         sig = librosa.effects.pitch_shift(sig_orig, self.sr, n_steps=i)
            #     else:
            #         sig = sig_orig
            #     self.data.append(sig)
            #     self.labels.append(label_tmp)

        self.random_noise, sr = librosa.load("./data/enviroment1.mp3", sr=self.sr, mono=True)

        print(len(self.filenames))


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # x = self.data[idx]
        x = np.load(self.filenames[idx])
        x = librosa.core.resample(x, 32000, self.sr)
        y = self.labels[idx]

        """prepare another sound for mixup augumentation"""
        # x2mix = self.data[random.randint(0,len(self.data)-1)]
        # x2mix = np.load(self.filenames[random.randint(0,len(self.filenames)-1)])
        # x2mix = librosa.core.resample(x2mix, 32000, self.sr)
        # y2mix = self.labels[random.randint(0,len(self.filenames)-1)]

        # if self.transform:
        """Random_slice"""
        # x, y, x2mix, y2mix = self.transform[0]([x, y, x2mix, y2mix]) # Random_slice_double

        x, y = self.transform[0](x, y)

        if len(self.transform) > 1 and False:
            """Mixup"""
            # x, y = self.transform[1]([x, y, x2mix, y2mix])
            """Random_Noise"""
            x, y = self.transform[2]([x, y, self.random_noise])
        # y = torch.LongTensor(np.argmax(y))
        # y = np.argmax(y)
        # print(x)
        """quantize the input or not"""
        # quantized_audio = quantize_data(x)
        # return quantized_audio, y
        return np.expand_dims(x, 0), y
        # return np.float32(x), np.argmax(y)



def clip_silence(audio_dir, files_list):

    """NEED TO ADD PATH -> export PATH=$PATH:/Users/taku-ueki/sox-14.4.2"""
    import os
    fin = open(files_list, "r")
    lines = fin.readlines()
    save_dir = Path("data/processed_data_wav")

    for i, line in enumerate(lines[1:]):
        filename = line.split(",")[0]
        file = str(audio_dir / filename)
        aug_cmd = "norm -0.1 silence 1 0.025 0.15% norm -0.1 reverse silence 1 0.025 0.15% reverse"
        aug_audio_file = str(save_dir / filename)
        os.system("sox %s %s %s" % (file, aug_audio_file, aug_cmd))
        # exit()
        if i % 100 == 0:
            print("{} / {}".format(i, len(lines)))

def convert2npy(data_dir, save_dir, sr=32000):
    from pathlib import Path
    data_dir = Path(data_dir)
    files = list(data_dir.glob('**/*.wav'))

    save_dir = Path(save_dir)
    for file in files:
        sig, sr = librosa.load(file, sr=sr, mono=True)
        filename = str(file.name).replace("wav", "npy")
        np.save(save_dir / filename, sig)

if __name__ == '__main__':
    # clip silence
    # clip_silence(Path("/Users/taku-ueki/Documents/data_set/freesound-audio-tagging/audio_train"), "/Users/taku-ueki/Documents/data_set/freesound-audio-tagging/train.csv")

    # convert wav to numpy array
    convert2npy("data/clipped_audio_wav", "data/clipped_audio_npy")
    exit()

    transform = transforms.Compose([Random_slice(), Mixup()])
    dataset = AudioDataset(Path("data/processed_data"), \
        "test_data_prep.txt", transform=[Random_slice(), Mixup(), Random_Noise()])

    print(dataset[0][0].shape)
    print(dataset[1][0].shape)
    print(dataset[2][0].shape)
