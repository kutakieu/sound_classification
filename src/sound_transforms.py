import numpy as np
import random

def padding(X, length):
    left = random.randint(0,length)
    right = length - left
    return np.pad(X, ((0,0), (left, right)), "constant")

class Random_slice_single():
    def __init__(self, min_frames=None, n_frames=3*128):
        self.min_frames = min_frames
        self.n_frames = n_frames

    def __call__(self, X_y):
        X, y = X_y
        if X.shape[1] < self.n_frames:
            X = padding(X, self.n_frames-X.shape[1])

        if self.min_frames is not None:
            frames = np.random.randint(self.min_frames, self.n_frames + 1)
        else:
            frames = self.n_frames

        # apply random cyclic shift
        max_idx = X.shape[1] - frames
        X_new = np.zeros((X.shape[0], frames), dtype=np.float32)
        start = np.random.randint(0, max_idx) if max_idx > 0 else 0
        stop = start + frames
        X_new = X[:, start:stop]

        return X_new, y

class Random_slice():
    def __init__(self, min_frames=None, n_frames=3*128):
        self.min_frames = min_frames
        self.n_frames = n_frames

    def __call__(self, X_y):
        X, y, X2mix, y2mix = X_y
        if X.shape[1] < self.n_frames:
            X = padding(X, self.n_frames-X.shape[1])
        if X2mix.shape[1] < self.n_frames:
            X2mix = padding(X2mix, self.n_frames-X2mix.shape[1])
        # print("Random_slice")
        # """padding! if the audio is too short"""
        # print(X.shape)
        # print(X2mix.shape)
        if self.min_frames is not None:
            frames = np.random.randint(self.min_frames, self.n_frames + 1)
        else:
            frames = self.n_frames

        # apply random cyclic shift
        max_idx = X.shape[1] - frames
        X_new = np.zeros((X.shape[0], frames), dtype=np.float32)
        start = np.random.randint(0, max_idx) if max_idx > 0 else 0
        stop = start + frames
        X_new = X[:, start:stop]

        max_idx = X2mix.shape[1] - frames
        X2mix_new = np.zeros((X2mix.shape[0], frames), dtype=np.float32)
        start = np.random.randint(0, max_idx) if max_idx > 0 else 0
        stop = start + frames
        X2mix_new = X2mix[:, start:stop]

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
        h, w = X.shape
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
    def __init__(self, type="wav"):
        self.type=type

    def __call__(self, X_y):
        X, y, noise = X_y
        if self.type == "wav":
            noise = noise[100:X.shape[1]+100]
        else:
            noise = noise[:, 100:X.shape[1]+100]
        alpha = random.uniform(0.0, 0.25)
        X = X * alpha + noise * (1.0 - alpha)
        return X, y
