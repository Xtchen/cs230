from glob import glob
import os

from torch import nn
import editdistance
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from constants import (
    CHAR_SET,
    TEST_DIR,
)


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
    count = 0
    while has_content:

        # Fetch every 5 frames.
        count += 1
        if count % 5 == 0:
            has_content, frame = cap.read()
            continue

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


def check_ratio(path):
    with open(path) as a:
        content = a.read()

        if not content:
            return False

        if float(content) < 0.9:
            print('{} skipped for low transferring ratio.'.format(path))
            return False

    return True


def load_testing_data():
    ret = []

    all_mp4 = sorted(glob(os.path.join(TEST_DIR, '*.mp4')))
    all_ratio = sorted(glob(os.path.join(TEST_DIR, '*.ratio')))

    for idx in range(len(all_mp4)):

        # Skip for invalid videoes.
        if not check_ratio(all_ratio[idx]):
            continue

        x = load_video(all_mp4[idx])
        ret.append((x, all_mp4[idx]))

    print('total valid data size: {}'.format(len(ret)))
    return ret


if __name__ == '__main__':
    # watcher = Watch()
    # try:
    #     # for some reason, the first attempt using cuda always fails for me...
    #     watcher = watcher.to('cuda' if torch.cuda.is_available() else 'cpu')
    # except:
    #     watcher = watcher.to('cuda' if torch.cuda.is_available() else 'cpu')
    # speller = Spell()
    # speller = speller.to('cuda' if torch.cuda.is_available() else 'cpu')

    watcher = torch.load('/home/ec2-user/modelStates/watch85.pt')
    speller = torch.load('/home/ec2-user/modelStates/spell85.pt')

    losses = []

    criterion = nn.CrossEntropyLoss()

    for x, path in load_testing_data():
        loss = 0
        guess = []

        x = x.to('cuda' if torch.cuda.is_available() else 'cpu')
        output_from_vgg_lstm, states_from_vgg_lstm = watcher(x)

        spell_input = torch.tensor([[CHAR_SET.index('<sos>')]]).repeat(output_from_vgg_lstm.size(0), 1).to('cuda' if torch.cuda.is_available() else 'cpu')
        spell_hidden = states_from_vgg_lstm
        spell_state = torch.zeros_like(spell_hidden).to('cuda' if torch.cuda.is_available() else 'cpu')
        context = torch.zeros(output_from_vgg_lstm.size(0), 1, spell_hidden.size(2)).to('cuda' if torch.cuda.is_available() else 'cpu')

        for idx in range(1000):
            spell_output, spell_hidden, spell_state, context = speller(spell_input, spell_hidden, spell_state, output_from_vgg_lstm, context)
            _, topi = spell_output.topk(1, dim=2)
            spell_input = topi.squeeze(1).detach()

            curr_predict = int(topi.squeeze(1)[0])
            guess.append(curr_predict)

            if CHAR_SET[curr_predict] in ['<eos>', '<sos>']:
                break

        prediction = ''
        try:
            for ii in range(len(guess)):
                prediction += CHAR_SET[int(guess[ii])]
            print(path)
            print(f'guess: {prediction}')
        except Exception as e:
            print(f'==================skip output================== due to error {e}')
