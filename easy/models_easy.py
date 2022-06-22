import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class CustomCNN(nn.Module):
    def __init__(self, cnn_hidden_size):
        super(CustomCNN, self).__init__()

        self.cnn_hidden_size = cnn_hidden_size

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2 * self.cnn_hidden_size, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(True)
        self.fc = nn.Linear(2 * self.cnn_hidden_size * 14 * 14, 64)

    def forward(self, inputs):
        """
        For reference (shape example)
        inputs: [Batch size x Sequence_length, Channel=1, Height, Width]
        outputs: [Sequence_length X Batch_size, Hidden_dim]
        """

        x = inputs.view(inputs.size(0) * inputs.size(1), 1, 28, 28)  # [BxS, C, H, W]
        out = self.maxpool(self.relu(self.conv1(x)))  # [BxS, 2*hid, 14, 14]
        out = out.view(-1, out.size(1) * out.size(2) * out.size(3))  # [BxS, 2*hid*14*14]
        outputs = self.relu(self.fc(out)).cuda()  # [BxS, 64]
        outputs = outputs.view(inputs.size(0), inputs.size(1), -1)
        return outputs


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, vocab_size, num_layers=1):
        super(LSTM, self).__init__()

        # define the properties
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        self.fc_in = nn.Linear(self.input_dim, self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, feature, h, c):
        """
        For reference (shape example)
        feature: (Sequence_length, Batch_size, Input_dim)
        """

        feature = self.fc_in(feature)
        output, (h_next, c_next) = self.lstm(feature, (h, c))
        output = self.fc_out(output)

        # (sequence_length, batch, num_classes), (num_rnn_layers, batch, hidden_dim), (num_rnn_layers, batch, hidden_dim)
        return output, h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, sequence_length=5, num_classes=26, cnn_layers=None,
                 cnn_input_dim=1, rnn_input_dim=256,
                 cnn_hidden_size=256, rnn_hidden_size=512, rnn_num_layers=1,
                 batch_size=256):
        super(ConvLSTM, self).__init__()

        # define the properties, you can freely modify or add hyperparameters
        self.cnn_hidden_size = cnn_hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.cnn_input_dim = cnn_input_dim
        self.rnn_input_dim = rnn_input_dim
        self.rnn_num_layers = rnn_num_layers
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.conv = CustomCNN(self.cnn_hidden_size)
        self.lstm = LSTM(self.rnn_input_dim, self.rnn_hidden_size, self.num_classes, self.rnn_num_layers)

    def forward(self, inputs):
        """
        input is (images, labels) (training phase) or images (test phase)
        images: sequential features of [Batch size, Sequence_length, Channel=1, Height, Width]
        labels: [Batch size, Sequence_length, Vocab_size]
        """

        if len(inputs) == 2:
            images, labels = inputs
        else:
            images = inputs

        images = torch.stack(images, dim=0).cuda()

        h0 = torch.zeros(self.rnn_num_layers, images.size(0), self.rnn_hidden_size).cuda()
        c0 = torch.zeros(self.rnn_num_layers, images.size(0), self.rnn_hidden_size).cuda()

        conv_out = self.conv(images)  # [S, B, 64]
        outputs, _, _ = self.lstm(conv_out, h0, c0)  # [S, B, 26]
        outputs = outputs.view(images.size(0), self.sequence_length, -1)

        return outputs
