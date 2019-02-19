from torch import nn
import torch

from constants import CHAR_SET


class VGG(nn.Module):
    """VGG-M: https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16/chung16.pdf.

    Modified the architecture a little. In the paper they consider 5 frames as a unit.
    Here, we start with a single frame. May need to tune it in the future.

    """

    def __init__(self):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, (7, 7), (2, 2)),  # _, 57, 57, 96
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0)),  # _, 28, 28, 96
            nn.Conv2d(96, 256, (5, 5), (2, 2), (1, 1)),  # _, 13, 13, 256
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0)),  # _, 6, 6, 256
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),  # _, 6, 6, 512
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),  # _, 6, 6, 512
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),  # _, 6, 6, 512
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True)  # _, 3, 3, 512 = _ * 4608
        )

        self.classifier = nn.Linear(4608, 512)

    def forward(self, x):
        """Expect x with shape (# of timestep, 5, 120, 120)"""
        x = self.features(x.view(-1, 1, 120, 120))  # # of timestep, 3, 3, 512

        x = x.view(x.size(0), -1)  # flatten to (_, 4608)
        x = self.classifier(x)  # (_, 512) mapping of frames to classes

        return x


class Watch(nn.Module):
    """Feed video to VGG and LSTM."""
    def __init__(self):
        super(Watch, self).__init__()
        self.vgg = VGG()
        self.lstm = nn.LSTM(512, 512, num_layers=3, batch_first=True)

    def forward(self, x):
        output_from_vgg = self.vgg(x).view(1, -1, 512)  # (# of timestep, 512)
        output_from_vgg_lstm, states_from_vgg_lstm = self.lstm(output_from_vgg)

        # output_from_vgg_lstm: (_, 1, 512)
        # states_from_vgg_lstm[0]: (3, _, 512)
        return output_from_vgg_lstm, states_from_vgg_lstm[0]


class Attention(nn.Module):
    """Reference: https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/
    The attention layer.

    """
    def __init__(self, hidden_size, annotation_size):
        super(Attention, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(hidden_size + annotation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, prev_hidden_state, annotations):
        batch_size, sequence_length = annotations.size(0), annotations.size(1)
        prev_hidden_state = prev_hidden_state.repeat(sequence_length, 1, 1).transpose(0, 1)
        concatenated = torch.cat([prev_hidden_state, annotations], dim=2)
        alpha = torch.nn.functional.softmax(self.dense(concatenated).squeeze(2)).unsqueeze(1)

        return alpha.bmm(annotations)


class Spell(nn.Module):
    """Reference: https://www.robots.ox.ac.uk/~vgg/publications/2017/Chung17/chung17.pdf"""
    def __init__(self):
        super(Spell, self).__init__()
        self.hidden_size = 512
        self.output_size = len(CHAR_SET)
        self.num_layers = 3

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size, self.num_layers, batch_first=True)
        self.attention = Attention(self.hidden_size, self.hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.hidden_size)
        )

    def forward(self, spell_input, hidden_state, spell_state, watcher_outputs, context):
        spell_input = self.embedding(spell_input)
        concatenated = torch.cat([spell_input, context], dim=2)
        output, (hidden_state, spell_state) = self.lstm(concatenated, (hidden_state, spell_state))
        context = self.attention(hidden_state[-1], watcher_outputs)
        output = self.mlp(torch.cat([output, context], dim=2).squeeze(1)).unsqueeze(1)

        return output, hidden_state, spell_state, context



