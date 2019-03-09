from glob import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from constants import (
    CHAR_SET,
    DEST_DIR,
    LEARNING_RATE,
    LEARNING_RATE_GAMMA,
    LEARNING_RATE_STEP,
)
from model import Watch, Spell


def plot_losses(losses):
    plt.xlabel('# iter')
    plt.ylabel('loss')
    plt.plot(range(len(losses)), losses)
    plt.show()


def load_video(video_path):
    """reshape the (# of frame, 120, 120) video to (batch size, # from frame, 120, 120)"""
    cap = cv2.VideoCapture(video_path)
    buffer = []

    has_content, frame = cap.read()
    while has_content:

        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (120, 120)).reshape(1, 120, 120)
        buffer.append(gray)
        has_content, frame = cap.read()

    try:
        results = torch.zeros(512, 120, 120)  # torch believes image_tensor is a BytesTensor. let's convert it to int.
        image_tensor = torch.from_numpy(np.concatenate(buffer, axis=0))
        # DataLoader wants size of tensors match.
        idx = torch.tensor([i for i in range(image_tensor.size(0) - 1, -1, -1)])
        results[:image_tensor.size(0), :, :] = image_tensor[idx, :, :]
    except Exception as e:
        print(f'due to error: {e}\nfailed to load video {video_path}')

    cap.release()

    return results


def load_one_entry(video_path, text_path):
    x = load_video(video_path)
    with open(text_path, 'r') as f:
        first_line = f.readline()
        first_line = first_line.replace(' ', '').rstrip('\n')
        chars = [CHAR_SET.index(i) for i in first_line.split(':')[1]]

        # add eos to the end.
        chars.append(CHAR_SET.index('<eos>'))

        # DataLoader wants size of tensors match.
        if len(chars) < 512:
            chars += [CHAR_SET.index('<pad>') for _ in range(512 - len(chars))]

        chars = torch.Tensor(chars)
    return x, chars


class MyDataSet(Dataset):
    """https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel"""
    def __init__(self):
        self.all_mp4 = sorted(glob(os.path.join(DEST_DIR, '*.mp4')))
        self.all_txt = sorted(glob(os.path.join(DEST_DIR, '*.txt')))

    def __len__(self):
        return len(self.all_mp4)

    def __getitem__(self, index):
        return load_one_entry(self.all_mp4[index], self.all_txt[index])


def data_loader():
    training_set = MyDataSet()
    train_len = len(training_set)
    indexes = list(range(train_len))
    np.random.shuffle(indexes)
    train_sampler = SubsetRandomSampler(indexes)

    train_loader = DataLoader(
        training_set, batch_size=32, sampler=train_sampler,
        num_workers=4, pin_memory=True
    )

    return train_loader


if __name__ == '__main__':
    watcher = Watch()
    try:
        # for some reason, the first attempt using cuda always fails for me...
        watcher = watcher.to('cuda' if torch.cuda.is_available() else 'cpu')
    except:
        watcher = watcher.to('cuda' if torch.cuda.is_available() else 'cpu')
    speller = Spell()
    speller = speller.to('cuda' if torch.cuda.is_available() else 'cpu')

    losses = []

    # Applying learning rate decay as we observed loss diverge.
    # https://discuss.pytorch.org/t/how-to-use-torch-optim-lr-scheduler-exponentiallr/12444/6
    watch_optimizer = optim.Adam(watcher.parameters(), lr=LEARNING_RATE)
    spell_optimizer = optim.Adam(speller.parameters(), lr=LEARNING_RATE)
    watch_scheduler = optim.lr_scheduler.StepLR(watch_optimizer, step_size=LEARNING_RATE_STEP, gamma=LEARNING_RATE_GAMMA)
    spell_scheduler = optim.lr_scheduler.StepLR(spell_optimizer, step_size=LEARNING_RATE_STEP, gamma=LEARNING_RATE_GAMMA)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(110):
        watch_scheduler.step()
        spell_scheduler.step()

        watcher = watcher.train()
        speller = speller.train()

        for i, (x, chars) in enumerate(data_loader()):
            loss = 0
            guess = []
            watch_optimizer.zero_grad()
            spell_optimizer.zero_grad()

            x = x.to('cuda' if torch.cuda.is_available() else 'cpu')
            chars = chars.to('cuda' if torch.cuda.is_available() else 'cpu')
            output_from_vgg_lstm, states_from_vgg_lstm = watcher(x)
            chars_len = chars.size(1)

            spell_input = torch.tensor([[CHAR_SET.index('<sos>')]]).repeat(output_from_vgg_lstm.size(0), 1).to('cuda' if torch.cuda.is_available() else 'cpu')
            spell_hidden = states_from_vgg_lstm
            spell_state = torch.zeros_like(spell_hidden).to('cuda' if torch.cuda.is_available() else 'cpu')
            context = torch.zeros(output_from_vgg_lstm.size(0), 1, spell_hidden.size(2)).to('cuda' if torch.cuda.is_available() else 'cpu')

            for idx in range(chars_len):
                spell_output, spell_hidden, spell_state, context = speller(spell_input, spell_hidden, spell_state, output_from_vgg_lstm, context)
                _, topi = spell_output.topk(1, dim=2)
                spell_input = chars[:, idx].long().unsqueeze(1)
                # import pdb; pdb.set_trace()
                loss += criterion(spell_output.squeeze(1), chars[:, idx].long())

                guess.append(int(topi.squeeze(1)[0]))

            if epoch % 5 == 0:
                label = ''
                prediction = ''
                try:
                    #import pdb; pdb.set_trace()
                    for ii in range(chars_len):
                        label += CHAR_SET[int(chars[0][ii])]
                        prediction += CHAR_SET[int(guess[ii])]
                    print(f'label: {label}')
                    print(f'guess: {prediction}')
                except Exception as e:
                    print(f'==================skip output================== due to error {e}')

        loss = loss.to('cuda' if torch.cuda.is_available() else 'cpu')
        loss.backward()
        watch_optimizer.step()
        spell_optimizer.step()

        norm_loss = float(loss / chars.size(0))
        losses.append(norm_loss)

        print(f'loss {norm_loss} epoch {epoch}')
    watcher = watcher.eval()
    speller = speller.eval()
    print(f'{losses}')
    plot_losses(losses)
