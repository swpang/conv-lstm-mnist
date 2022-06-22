import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class CustomCNN(nn.Module):
    def __init__(self, rnn_input_dim, block, cnn_layers):
        super(CustomCNN, self).__init__()

        self.in_channels = 16
        self.rnn_input_dim = rnn_input_dim

        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(True)
        self.layer1 = self.make_layer(block, 16, cnn_layers[0])
        self.layer2 = self.make_layer(block, 32, cnn_layers[1], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(32, self.rnn_input_dim)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, inputs):
        """
        For reference (shape example)
        inputs: [Batch size x Sequence_length, Channel=1, Height, Width]
        outputs: [Sequence_length X Batch_size, Hidden_dim]
        """

        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        x = inputs.view(seq_len, 1, 28, 28)  # [BxS, C, H, W]

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.avg_pool(out)
        out = out.view(batch_size, seq_len, -1)
        outputs = self.fc(out)

        return outputs


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, vocab_size, num_layers=1, bidirectional=True):
        super(LSTM, self).__init__()

        # define the properties
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True, bidirectional=self.bidirectional)
        self.fc_in = nn.Linear(self.input_dim, self.hidden_size)
        if self.bidirectional:
            self.fc_out = nn.Linear(2 * self.hidden_size, self.vocab_size)
        else:
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
    def __init__(self, rnn_input_dim, rnn_hidden_size, rnn_num_layers,
                 batch_size, cnn_layers, bidirectional, dropout, num_classes=26):
        super(ConvLSTM, self).__init__()

        # define the properties, you can freely modify or add hyperparameters
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_input_dim = rnn_input_dim
        self.rnn_num_layers = rnn_num_layers
        self.cnn_layers = cnn_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.batch_size = batch_size

        self.conv = CustomCNN(self.rnn_input_dim, ResidualBlock, self.cnn_layers)
        self.lstm = LSTM(self.rnn_input_dim, self.rnn_hidden_size, self.num_classes,
                         self.rnn_num_layers, self.bidirectional, self.dropout)

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

        if self.bidirectional:
            k = 2
        else:
            k = 1

        outputs_l = []
        for x in images:
            x = x.cuda()
            x = x.unsqueeze(0)  # batch size [1, Seq, C, H, W]
            # print(x.shape)

            h0 = torch.zeros(self.rnn_num_layers * k, 1, self.rnn_hidden_size).cuda()
            c0 = torch.zeros(self.rnn_num_layers * k, 1, self.rnn_hidden_size).cuda()

            conv_out = self.conv(x)  # [1, Seq, rnn_input]
            # print(conv_out.shape)
            rnn_out, _, _ = self.lstm(conv_out, h0, c0)  # [1, Seq, 26]
            # print(output.shape)
            output = rnn_out[:, -1, :].view(self.num_classes)
            outputs_l.append(output)
        outputs = torch.stack(outputs_l, dim=0).cuda()

        return outputs
