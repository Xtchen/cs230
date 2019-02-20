from glob import glob
import os

from torch import nn, optim
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

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
        buffer = buffer[:int(len(buffer)/5)*5]  # make sure the total frame can be dived by 5 by dropping the extra frames.
        image_tensor = torch.from_numpy(np.concatenate(buffer, axis=0))
        results = torch.zeros(len(buffer), 120, 120)  # torch believes image_tensor is a BytesTensor. let's convert it to int.
        results[:, :, :] = image_tensor[:, :, :]
        results = results.view(1, -1, 120, 120)
    except Exception as e:
        print(f'due to error: {e}\nfailed to load video {video_path}')

    cap.release()

    return results


def load_training_data():
    ret = []
    all_mp4 = sorted(glob(os.path.join(DEST_DIR, '*.mp4')))
    all_txt = sorted(glob(os.path.join(DEST_DIR, '*.txt')))

    assert len(all_mp4) == len(all_txt)

    for idx in range(len(all_mp4)):
        x = load_video(all_mp4[idx])
        with open(all_txt[idx], 'r') as f:
            first_line = f.readline()
            first_line = first_line.replace(' ', '').rstrip('\n')
            chars = [CHAR_SET.index(i) for i in first_line.split(':')[1]]

            # add eos to the end.
            chars.append(CHAR_SET.index('<eos>'))
            chars = torch.Tensor(chars)
        ret.append((x, chars))

    return ret


if __name__ == '__main__':
    watcher = Watch()
    try:
        # for some reason, the first attempt using cuda always fails for me...
        watcher = watcher.to('cuda')
    except:
        watcher = watcher.to('cuda')
    speller = Spell()
    speller = speller.to('cuda')

    losses = []

    # Applying learning rate decay as we observed loss diverge.
    # https://discuss.pytorch.org/t/how-to-use-torch-optim-lr-scheduler-exponentiallr/12444/6
    watch_optimizer = optim.Adam(watcher.parameters(), lr=LEARNING_RATE)
    spell_optimizer = optim.Adam(speller.parameters(), lr=LEARNING_RATE)
    watch_scheduler = optim.lr_scheduler.StepLR(watch_optimizer, step_size=LEARNING_RATE_STEP, gamma=LEARNING_RATE_GAMMA)
    spell_scheduler = optim.lr_scheduler.StepLR(spell_optimizer, step_size=LEARNING_RATE_STEP, gamma=LEARNING_RATE_GAMMA)

    criterion = nn.CrossEntropyLoss(ignore_index=CHAR_SET.index('<pad>'))

    for epoch in range(110):
        watch_scheduler.step()
        spell_scheduler.step()

        watcher = watcher.train()
        speller = speller.train()

        for (x, chars) in load_training_data():
            loss = 0
            guess = []
            watch_optimizer.zero_grad()
            spell_optimizer.zero_grad()

            x = x.to('cuda')
            chars = chars.to('cuda')
            output_from_vgg_lstm, states_from_vgg_lstm = watcher(x)
            chars_len = chars.size(0)

            spell_input = torch.tensor([[CHAR_SET.index('<sos>')]]).repeat(output_from_vgg_lstm.size(0), 1).to('cuda')
            spell_hidden = states_from_vgg_lstm
            spell_state = torch.zeros_like(spell_hidden).to('cuda')
            context = torch.zeros(output_from_vgg_lstm.size(0), 1, spell_hidden.size(2)).to('cuda')

            for idx in range(chars_len):
                spell_output, spell_hidden, spell_state, context = speller(spell_input, spell_hidden, spell_state, output_from_vgg_lstm, context)
                _, topi = spell_output.topk(1, dim=2)
                spell_input = chars[idx].long().view(1, 1)
                # import pdb; pdb.set_trace()
                loss += criterion(spell_output.squeeze(1), chars[idx].long().view(1))

                # print(f'truth char: {chars[idx]} | guess char: {int(topi.squeeze(1)[0])}')
                guess.append(int(topi.squeeze(1)[0]))

            if epoch % 5 == 0:
                label = ''
                prediction = ''
                try:
                    for ii in range(chars_len):
                        label += CHAR_SET[int(chars[ii])]
                        prediction += CHAR_SET[int(guess[ii])]
                    print(f'label: {label}')
                    print(f'guess: {prediction}')
                except Exception as e:
                    print(f'==================skip output================== due to error {e}')

            loss = loss.to('cuda')
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