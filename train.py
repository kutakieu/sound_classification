import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import pickle
from pathlib import Path
import numpy as np

from src.model.VGG import Sound_VGG
from src.model.WaveNet import WaveNet
from src.model.WaveNet4classification import WaveNet4classification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_softmax = nn.LogSoftmax(dim=1)
def loss_fn_mixup(x,y):
    loss = -log_softmax(x) * y
    return loss.mean()

def main(args):
    print("device : {}".format(device))
    print(args)
    data_dir = Path(args.data_dir)

    if args.model == "vgg":
        from src.dataset_spectrogram_invertable import AudioDataset
        from src.sound_transforms import Random_slice_single, Mixup, Random_Noise

        model = Sound_VGG().to(device)
        transforms_train = [Random_slice_single(), Mixup(), Random_Noise()]
        transforms_test = [Random_slice_single()]

    elif args.model == "wavenet":
        from src.dataset_wav import AudioDataset
        from src.sound_transforms_wav import Random_slice, Mixup, Random_Noise, Random_slice_fixed_length

        model = WaveNet4classification(batch_size=args.batch_size, n_layers=4, n_blocks=6).to(device)
        print("receptive_field: " + str(model.receptive_field))
        if args.batch_size == 1:
            transforms_train = [Random_slice(min_frames=model.receptive_field, max_frames=model.receptive_field+24000), Mixup(), Random_Noise()]
            transforms_test = [Random_slice(min_frames=model.receptive_field, max_frames=model.receptive_field+24000)]
        else:
            transforms_train = [Random_slice_fixed_length(receptive_field=model.receptive_field), Mixup(), Random_Noise()]
            transforms_test = [Random_slice_fixed_length(receptive_field=model.receptive_field)]


    """dataset for training"""
    if Path('./data/dataset_training.pkl').exists():
        # with open('./data/dataset_training.pkl', mode='rb') as f:
        with open('./data/dataset_training.pkl', mode='rb') as f:
            dataset_training = pickle.load(f)
    else:
        if args.debug:
            print("debug mode")
            dataset_training = AudioDataset(data_dir, "./data/debug/train.txt", transform=transforms_train)
        else:
            dataset_training = AudioDataset(data_dir, "./data/train.txt", transform=transforms_train)
        # with open('./data/dataset_training.pkl', mode='wb') as f:
        #     pickle.dump(dataset_training, f)
    dataloader_training = torch.utils.data.DataLoader(dataset_training, batch_size=args.batch_size, shuffle=True)

    """dataset for test"""
    if Path('./data/dataset_test.pkl').exists():
        with open('./data/dataset_test.pkl', mode='rb') as f:
            dataset_test = pickle.load(f)
    else:
        if args.debug:
            dataset_test = AudioDataset(data_dir, "./data/debug/test.txt", transform=transforms_test)
        else:
            dataset_test = AudioDataset(data_dir, "./data/test.txt", transform=transforms_test)
        # with open('./data/dataset_test.pkl', mode='wb') as f:
        #     pickle.dump(dataset_test, f)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)



    # model.to(device)
    if args.loss_fn == "CE":
        loss_function = nn.CrossEntropyLoss()
    elif args.loss_fn == "BCE":
        # loss_function = nn.BCELoss()
        loss_function = nn.functional.binary_cross_entropy_with_logits
        # loss_function = loss_fn_mixup
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    softmax = nn.Softmax(dim=1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)


    """training"""
    model.train()
    loss_history_training = []
    accuracy_history_training = []
    loss_history_validation = []
    accuracy_history_validation = []
    for epoch in range(args.epochs):
        loss_history_training.append(0.0)
        accuracy_history_training.append(0)
        scheduler.step()
        for i, batch in enumerate(dataloader_training):
            X = batch[0]
            if X.shape[0] == 1:
                continue
            if args.model == "vgg":
                X = torch.unsqueeze(X, 1).to(device)
            else:
                X = X.to(device)
            y = batch[1].to(device)
            logit = model(X, is_training=True)
            prediction = torch.argmax(logit, 1).squeeze()
            # print("logit.shape = {}".format(logit.shape))
            if args.loss_fn == "CE":
                logit = logit.squeeze()
                # print(logit.shape)
                # print(y)
                loss = loss_function(logit, y)
            elif args.loss_fn == "BCE":
                # softmax_logit = softmax(logit)
                loss = loss_function(logit, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history_training[epoch] += loss.item()
            if args.loss_fn == "CE":
                correct = torch.eq(y, prediction).sum().item()
            elif args.loss_fn == "BCE":
                correct = torch.eq(torch.argmax(y), prediction).sum().item()

            if i % 20 == 0:
                print("loss = {}".format(loss))
            # print("accuracy = {}".format(correct))

            accuracy_history_training[epoch] += correct

            # if epoch % 10 == 0:
            #     print(loss)
        accuracy_history_training[epoch] /= float(len(dataset_training))
        loss_history_training[epoch] /= float(len(dataset_training))

        """validation"""
        model.eval()
        print("epoch {} validation".format(epoch))
        prediction_list = []
        label_list = []
        with torch.no_grad():
            loss_history_validation.append(0.0)
            accuracy_history_validation.append(0)
            for batch in dataloader_test:
                # print(batch[0].shape)
                X = batch[0]
                if X.shape[0] == 1:
                    continue
                if args.model == "vgg":
                    X = torch.unsqueeze(X, 1).to(device)
                else:
                    X = X.to(device)

                y = batch[1].to(device)

                logit = model(X, is_training=False)
                prediction = torch.argmax(logit, 1).squeeze()
                if args.loss_fn == "CE":
                    loss = loss_function(logit.squeeze(), y)
                elif args.loss_fn == "BCE":
                    loss = loss_function(logit, y)

                loss_history_validation[epoch] += loss.item()
                if args.loss_fn == "CE":
                    correct = torch.eq(y, prediction).sum().item()
                elif args.loss_fn == "BCE":
                    correct = torch.eq(torch.argmax(y), prediction).sum().item()
                accuracy_history_validation[epoch] += correct

                prediction = list(prediction.detach().cpu().numpy())
                prediction_list += prediction
                labels = list(y.detach().cpu().numpy())
                label_list += labels

            accuracy_history_validation[epoch] /= float(len(dataset_test))
            loss_history_validation[epoch] /= float(len(dataset_test))
            if epoch > 0 and accuracy_history_validation[epoch-1] < accuracy_history_validation[epoch]:
                torch.save(model.state_dict(), "./trained_model/epoch_{}.ckpt".format(epoch))
                np.save("prediction_list.npy", prediction_list)
                np.save("label_list.npy", label_list)


        print("loss_history_training = " + str(loss_history_training))
        print("accuracy_history_training = " + str(accuracy_history_training))
        print("loss_history_validation = " + str(loss_history_validation))
        print("accuracy_history_validation = " + str(accuracy_history_validation))





def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Attention based model for solving the Travelling Salesman Problem with Reinforcement Learning")

    # Data
    parser.add_argument('--batch_size', type=int, default=32, help='Number of instances per batch during training')
    parser.add_argument('--val_dataset', type=str, default=None, help='Dataset file to use for validation')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--pretrained_model', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--loss_fn', type=str, default="CE", help='Loss function default:CrossEntropyLoss')
    parser.add_argument('--debug', action="store_true", help='if it is in debug mode')
    parser.add_argument('--model', type=str, default="wavenet", help='type of model defalt:wavenet')
    parser.add_argument('--data_dir', type=str, default=None, help='path to the data directory', required=True)

    return parser.parse_args(args)

if __name__ == '__main__':
    main(get_options())
