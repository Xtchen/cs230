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
    count = 0
    while has_content:

        # Fetch every 3 frames.
        count += 1
        if count % 3 == 0:
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


def load_training_data():
    ret = []

    dirs = os.listdir(DEST_DIR)

    for dir in dirs:
        print('loading {}'.format(dir))

        all_mp4 = sorted(glob(os.path.join(DEST_DIR, dir, '*.mp4')))
        all_txt = sorted(glob(os.path.join(DEST_DIR, dir, '*.txt')))
        all_ratio = sorted(glob(os.path.join(DEST_DIR, dir, '*.ratio')))

        assert len(all_mp4) == len(all_txt)

        for idx in range(len(all_mp4)):

            # Skip for invalid videoes.
            if not check_ratio(all_ratio[idx]):
                continue

            x = load_video(all_mp4[idx])
            with open(all_txt[idx], 'r') as f:
                try:
                    first_line = f.readline()
                    first_line = first_line.replace(' ', '').rstrip('\n')
                    chars = [CHAR_SET.index(i) for i in first_line.split(':')[1]]

                    # add eos to the end.
                    chars.append(CHAR_SET.index('<eos>'))
                    chars = torch.Tensor(chars)
                except Exception as e:
                    print('Wrong text data, skip! {}: {}'.format(first_line, e))
                    continue
            ret.append((x, chars))

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

    watcher = torch.load('/home/ec2-user/modelStates/watch10.pt')
    speller = torch.load('/home/ec2-user/modelStates/spell10.pt')

    losses = []

    # Applying learning rate decay as we observed loss diverge.
    # https://discuss.pytorch.org/t/how-to-use-torch-optim-lr-scheduler-exponentiallr/12444/6
    watch_optimizer = optim.Adam(watcher.parameters(), lr=LEARNING_RATE)
    spell_optimizer = optim.Adam(speller.parameters(), lr=LEARNING_RATE)
    watch_scheduler = optim.lr_scheduler.StepLR(watch_optimizer, step_size=LEARNING_RATE_STEP, gamma=LEARNING_RATE_GAMMA)
    spell_scheduler = optim.lr_scheduler.StepLR(spell_optimizer, step_size=LEARNING_RATE_STEP, gamma=LEARNING_RATE_GAMMA)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(11, 110):
        watch_scheduler.step()
        spell_scheduler.step()

        # watcher = watcher.train()
        # speller = speller.train()

        with open('/home/ec2-user/modelStates/loss{}'.format(epoch), 'w+') as loss_writer:

            for (x, chars) in load_training_data():
                loss = 0
                guess = []
                watch_optimizer.zero_grad()
                spell_optimizer.zero_grad()

                x = x.to('cuda' if torch.cuda.is_available() else 'cpu')
                chars = chars.to('cuda' if torch.cuda.is_available() else 'cpu')
                output_from_vgg_lstm, states_from_vgg_lstm = watcher(x)
                chars_len = chars.size(0)

                spell_input = torch.tensor([[CHAR_SET.index('<sos>')]]).repeat(output_from_vgg_lstm.size(0), 1).to('cuda' if torch.cuda.is_available() else 'cpu')
                spell_hidden = states_from_vgg_lstm
                spell_state = torch.zeros_like(spell_hidden).to('cuda' if torch.cuda.is_available() else 'cpu')
                context = torch.zeros(output_from_vgg_lstm.size(0), 1, spell_hidden.size(2)).to('cuda' if torch.cuda.is_available() else 'cpu')

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

                loss = loss.to('cuda' if torch.cuda.is_available() else 'cpu')
                loss.backward()
                watch_optimizer.step()
                spell_optimizer.step()

                norm_loss = float(loss / chars.size(0))
                losses.append(norm_loss)

                print(f'loss {norm_loss} epoch {epoch}')
                loss_writer.write(f'{norm_loss} ')

        watcher = watcher.eval()
        speller = speller.eval()

        # Starting from epoch 11, save model's states.
        torch.save(watcher.state_dict(), '/home/ec2-user/modelStates/watch{}.pt'.format(epoch))
        torch.save(speller.state_dict(), '/home/ec2-user/modelStates/spell{}.pt'.format(epoch))

    print(f'{losses}')
    plot_losses(losses)
