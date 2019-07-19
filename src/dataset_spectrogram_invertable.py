import torch
from torch.utils.data import Dataset
# from torchvision import transforms
import librosa
from librosa.effects import pitch_shift
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random

from src.sound_transforms import Random_slice, Mixup, Random_Noise
from src.feature.utils import *
from src.feature.spectrograms_inversion import *


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

        self.fft_size = 2048  # window size for the FFT
        self.step_size = self.fft_size // 16  # distance to slide along the window (in time)
        self.spec_thresh = 4  # threshold for spectrograms (lower filters out more noise)
        self.lowcut = 500  # Hz # Low cut for our butter bandpass filter
        self.highcut = 15000  # Hz # High cut for our butter bandpass filter
        # For mels
        self.n_mel_freq_components = 128  # number of mel frequency channels
        self.shorten_factor = 1  # how much should we compress the x-axis (time)
        self.start_freq = 300  # Hz # What frequency to start sampling our melS from
        self.end_freq = 8000  # Hz # What frequency to stop sampling our melS from

        self.transform = transform


        self.labels = []
        self.filenames = []
        self.filename2label = {}

        files = list(data_dir.glob('**/*.npy'))
        files = [file.name for file in files]

        fin = open(data_list_file, "r")
        lines = fin.readlines()

        for line in lines[1:]:
            filename, label = line.split(",")[:2]
            filename = filename.replace(".wav", ".npy")
            if filename in files:
                self.filenames.append(data_dir / filename)
                # self.labels.append(np.array(classes.index(label)))
                self.filename2label[data_dir / filename] = np.array(classes.index(label))

            # """apply pitch_shift -3~3"""
            # for i in range(-3,3):
            #     spectrogram = self.preprocess(data_dir / filename, pitch_shift=i)
            #     if i != 0:
            #         sig = librosa.effects.pitch_shift(sig_orig, self.sr, n_steps=i)
            #     else:
            #         sig = sig_orig
            #     self.data.append(sig)
            #     self.labels.append(label_tmp)

        # self.random_noise, sr = librosa.load("./data/enviroment1.mp3", sr=self.sr, mono=True)
        mel_filter, mel_inversion_filter = create_mel_filter(   fft_size=self.fft_size,
                                                                n_freq_components=self.n_mel_freq_components,
                                                                start_freq=self.start_freq,
                                                                end_freq=self.end_freq,
                                                                samplerate = self.sr )
        data, signal_ratio = librosa.load("./data/enviroment1.mp3", sr=self.sr, mono=True)
        wav_spectrogram = pretty_spectrogram(
                                                data.astype("float64"),
                                                fft_size=self.fft_size,
                                                step_size=self.step_size,
                                                log=True,
                                                thresh=self.spec_thresh,
                                            )
        self.random_noise = make_mel(wav_spectrogram, mel_filter, shorten_factor=self.shorten_factor)

        # print(self.filenames[:10])
        # print(self.labels[:10])


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        x = np.load(self.filenames[idx])
        # y = self.labels[idx]
        y = self.filename2label[self.filenames[idx]]

        # x2mix = self.filenames[random.randint(0,len(self.filenames)-1)]
        # y2mix = self.labels[random.randint(0,len(self.filenames)-1)]
        # if self.transform:
        """Random_slice"""
        x, y = self.transform[0]([x, y])
        if random.random() > 0.5:
            noise_src, _ = self.transform[0]([self.random_noise, y])
            noise_mix_rate = random.uniform(0.025, 0.2)
            x = x*(1.0 - noise_mix_rate) + noise_src*noise_mix_rate

        return np.float32(x), y

        if len(self.transform) > 1:
            """Mixup"""
            x, y = self.transform[1]([x, y, x2mix, y2mix])
            """Random_Noise"""
            x, y = self.transform[2]([x, y, self.random_noise])
        # y = torch.LongTensor(np.argmax(y))
        # y = np.argmax(y)
        # print(x)
        return np.float32(x), y



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

def invert_mel_spectrogram(mel_spec, mel_inversion_filter):
    mel_inverted_spectrogram = mel_to_spectrogram(
                                                    mel_spec,
                                                    mel_inversion_filter,
                                                    spec_thresh=spec_thresh,
                                                    shorten_factor=shorten_factor,
                                                )
    inverted_mel_audio = invert_pretty_spectrogram(
                                                        np.transpose(mel_inverted_spectrogram),
                                                        fft_size=fft_size,
                                                        step_size=step_size,
                                                        log=True,
                                                        n_iter=10,
                                                    )
    return inverted_mel_audio



def convert2mel_spectrogram(data_dir, save_dir):

    # ### Parameters ###
    signal_ratio=16000
    fft_size = 2048  # window size for the FFT
    step_size = fft_size // 16  # distance to slide along the window (in time)
    spec_thresh = 4  # threshold for spectrograms (lower filters out more noise)
    lowcut = 500  # Hz # Low cut for our butter bandpass filter
    highcut = 15000  # Hz # High cut for our butter bandpass filter
    # For mels
    n_mel_freq_components = 128  # number of mel frequency channels
    shorten_factor = 1  # how much should we compress the x-axis (time)
    start_freq = 300  # Hz # What frequency to start sampling our melS from
    end_freq = 8000  # Hz # What frequency to stop sampling our melS from

    mel_filter, mel_inversion_filter = create_mel_filter(   fft_size=fft_size,
                                                            n_freq_components=n_mel_freq_components,
                                                            start_freq=start_freq,
                                                            end_freq=end_freq,
                                                            samplerate = signal_ratio )
    from pathlib import Path
    data_dir = Path(data_dir)
    files = list(data_dir.glob('**/*.wav'))

    save_dir = Path(save_dir)
    # for file_idx in tqdm(range(len(files))):
    for file in tqdm(files):
        try:
            # file = files[file_idx]
            data, signal_ratio = librosa.load(file, sr=signal_ratio, mono=True)
            wav_spectrogram = pretty_spectrogram(
                                                    data.astype("float64"),
                                                    fft_size=fft_size,
                                                    step_size=step_size,
                                                    log=True,
                                                    thresh=spec_thresh,
                                                )
            mel_spec = make_mel(wav_spectrogram, mel_filter, shorten_factor=shorten_factor)

            filename = str(file.name).replace("wav", "npy")
            np.save(save_dir / filename, mel_spec)
        except:
            print(file)



if __name__ == '__main__':
    # clip silence
    # clip_silence(Path("/Users/taku-ueki/Documents/data_set/freesound-audio-tagging/audio_train"), "/Users/taku-ueki/Documents/data_set/freesound-audio-tagging/train.csv")

    # convert wav to numpy array
    # convert2npy("data/clipped_audio_wav", "data/clipped_audio_npy")
    # exit()
    #
    # transform = transforms.Compose([Random_slice(), Mixup()])
    # dataset = AudioDataset(Path("data/processed_data"), \
    #     "test_data_prep.txt", transform=[Random_slice(), Mixup(), Random_Noise()])
    #
    # print(dataset[0][0].shape)
    # print(dataset[1][0].shape)
    # print(dataset[2][0].shape)

    #
    convert2mel_spectrogram("data/clipped_audio_wav", "data/invertable_mel_specs")
