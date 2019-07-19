import numpy as np
import random

def padding(X, length):
    left = random.randint(0,length)
    right = length - left
    return np.pad(X, (left, right), "constant", constant_values=(128,128))

def pad4receptive_field(X, receptive_field):
    return np.pad(X, (receptive_field, 0), "constant", constant_values=(0,0))

class Random_slice():
    def __init__(self, min_frames, max_frames, fix_length=False):
        self.min_frames = min_frames
        self.max_frames = max_frames


    def __call__(self, X, y):
        if X.shape[0] < self.min_frames:
            X_new = X
        else:
            frames = np.random.randint(self.min_frames, min(self.max_frames, X.shape[0]))
            max_idx = X.shape[0] - frames
            X_new = np.zeros((1, frames), dtype=np.float32)
            start = np.random.randint(0, max_idx) if max_idx > 0 else 0
            stop = start + frames
            X_new = X[start:stop]
        y_new = np.repeat(y, X_new.shape[0]+1, axis=0)
        X_new = pad4receptive_field(X_new, self.min_frames)

        return X_new, y_new

class Random_slice_fixed_length():
    def __init__(self, receptive_field, frames=40000):
        self.frames = frames
        self.receptive_field = receptive_field

    def __call__(self, X, y):
        X_new = np.zeros((self.frames), dtype=np.float32)
        if X.shape[0] < self.frames:
            for i in range(int(self.frames/X.shape[0])):
                X_new[X.shape[0]*i:X.shape[0]*(i+1)] = X
            X_new[-X.shape[0]:] = X
        else:
            max_idx = X.shape[0] - self.frames
            start = np.random.randint(0, max_idx) if max_idx > 0 else 0
            stop = start + self.frames
            X_new = X[start:stop]
        y_new = np.repeat(y, self.frames-self.receptive_field+1, axis=0)
        # y_new = np.repeat(y, X_new.shape[0]+1, axis=0)
        # X_new = pad4receptive_field(X_new, self.receptive_field)
        return X_new, y_new

class Random_slice_double():
    def __init__(self, n_frames, min_frames=None):
        self.min_frames = min_frames
        self.n_frames = n_frames

    def __call__(self, X_y):
        X, y, X2mix, y2mix = X_y
        if X.shape[0] < self.n_frames:
            X = padding(X, self.n_frames-X.shape[0])
        if X2mix.shape[0] < self.n_frames:
            X2mix = padding(X2mix, self.n_frames-X2mix.shape[0])
        # print("Random_slice")
        # """padding! if the audio is too short"""
        # print(X.shape)
        # print(X2mix.shape)
        if self.min_frames is not None:
            frames = np.random.randint(self.min_frames, self.n_frames + 1)
        else:
            frames = self.n_frames

        # apply random cyclic shift
        max_idx = X.shape[0] - frames
        X_new = np.zeros((1, frames), dtype=np.float32)
        start = np.random.randint(0, max_idx) if max_idx > 0 else 0
        stop = start + frames
        X_new = X[start:stop]

        max_idx = X2mix.shape[0] - frames
        X2mix_new = np.zeros((1, frames), dtype=np.float32)
        start = np.random.randint(0, max_idx) if max_idx > 0 else 0
        stop = start + frames
        X2mix_new = X2mix[start:stop]

        return X_new, y, X2mix_new, y2mix

class Mixup():
    def __init__(self, mix_label=False, multi_label=False):
        self.to_one_hot=True
        self.mix_label=mix_label
        self.multi_label=multi_label
        # self.alpha=None

    def __call__(self, X_y):
        X, y, x2mix, y2mix = X_y
        # print("Mixup")
        # h, w = X.shape
        # l = np.random.beta(self.alpha, self.alpha, batch_size)
        # X_l = l.reshape(batch_size, 1, 1, 1)
        # y_l = l.reshape(batch_size, 1)

        # mix observations
        # X1, X2 = X[:], X[::-1]
        alpha = random.uniform(0.1, 0.25)
        X = X * (1.0 - alpha)  + x2mix * alpha

        # mix labels
        if self.mix_label:
            y = y * (1.0 - alpha) + y2mix * alpha
        # to assign multiple labels
        if self.multi_label:
            y = y if np.array_equal(y, y2mix) else y + y2mix

        return X.astype(np.float32), y.astype(np.float32)

class Random_Noise():
    def __init__(self):
        self.type=type

    def __call__(self, X_y):
        X, y, noise = X_y
        noise = noise[100:X.shape[0]+100]
        alpha = random.uniform(0.0, 0.25)
        X = X * alpha + noise * (1.0 - alpha)
        return X, y
