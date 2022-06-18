import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def weights_init(m):
    torch.nn.init.xavier_uniform_(m.weight)

class CustomCNN(nn.Module):
    def __init__(self, cnn_input_size, cnn_hidden_size, rnn_hidden_size):
        # NOTE: you can freely add hyperparameters argument
        super(CustomCNN, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem1-1: define cnn model        
        # ResNet

        self.cnn_input_size = cnn_input_size
        self.cnn_hidden_size = cnn_hidden_size
        self.rnn_hidden_size = rnn_hidden_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=cnn_input_size, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.resblock1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=cnn_hidden_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=cnn_hidden_size, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.resblock2 = nn.Sequential(
            nn.Conv2d(in_channels=cnn_hidden_size, out_channels=cnn_hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=cnn_hidden_size, momentum=0.9),
            nn.ReLU(True),
            nn.Conv2d(in_channels=cnn_hidden_size, out_channels=cnn_hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=cnn_hidden_size, momentum=0.9),
            nn.ReLU(True)
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(True)

        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, inputs):
        """
        For reference (shape example)
        inputs: Batch size X (Sequence_length, Channel=1, Height, Width) -> this is a list
        outputs: (Sequence_length X Batch_size, Hidden_dim) -> this is a list
        """
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem1-2: code CNN forward path

        self.conv1.apply(weights_init)
        self.conv2.apply(weights_init)
        self.conv3.apply(weights_init)
        self.conv4.apply(weights_init)
        self.resblock1.apply(weights_init)
        self.resblock2.apply(weights_init)

        outputs = []

        # Iterate on batches
        for x in inputs:
            out = self.conv2(self.conv1(x))
            residual = out
            out = self.resblock1(out) + residual
            out = self.conv4(self.conv3(out))
            residual = out
            out = self.resblock2(out) + residual # out = [S, C, H, W]
            out = out.view(-1, out.size(1) * out.size(2) * out.size(3))

            fc1 = nn.Linear(out.size(1), self.rnn_hidden_size).cuda()

            outputs.append(self.relu(fc1(out)))

        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return outputs


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, vocab_size, num_layers=1, dropout=0):
        super(LSTM, self).__init__()

        # define the properties
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout = dropout

        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem2-1: Define lstm and input, output projection layer to fit dimension
        # output fully connected layer to project to the size of the class

        # you can either use torch LSTM or manually define it
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout)
        self.fc_in = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size, self.vocab_size)
        self.relu = nn.ReLU(True)

        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, feature, h, c):
        """
        For reference (shape example)
        feature: (Sequence_length, Batch_size, Input_dim)
        """
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem2-2: Design LSTM model for letter sorting
        # NOTE: sequence length of feature can be various

        feature = self.fc_in(feature)
        output, h_next, c_next = self.lstm(feature, (h, c))
        output = self.relu(self.fc_out(self.relu(output)))

        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

        # (sequence_length, batch, num_classes), (num_rnn_layers, batch, hidden_dim), (num_rnn_layers, batch, hidden_dim)
        return output, h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, sequence_length=5, num_classes=26, cnn_layers=None,
                 cnn_input_dim=1, rnn_input_dim=256,
                 cnn_hidden_size=256, rnn_hidden_size=512, rnn_num_layers=1,
                 rnn_dropout=0, teacher_forcing=False, batch_size=256):
        # NOTE: you can freely add hyperparameters argument
        super(ConvLSTM, self).__init__()

        # define the properties, you can freely modify or add hyperparameters
        self.cnn_hidden_size = cnn_hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.cnn_input_dim = cnn_input_dim
        self.rnn_input_dim = rnn_input_dim
        self.rnn_num_layers = rnn_num_layers
        self.rnn_dropout = rnn_dropout
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.teacher_forcing = teacher_forcing
        self.batch_size = batch_size
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        self.conv = CustomCNN(self.cnn_input_dim, self.cnn_hidden_size, self.rnn_hidden_size)
        self.lstm = LSTM(self.rnn_input_dim, self.rnn_hidden_size, self.rnn_num_layers, self.rnn_dropout)
        # NOTE: you can define additional parameters
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, inputs):
        """
        input is (images, labels) (training phase) or images (test phase)
        images: sequential features of Batch size X (Sequence_length, Channel=1, Height, Width)
        labels: Batch size X (Sequence_length)
        outputs should be a size of Batch size X (1, Num_classes) or Batch size X (Sequence_length, Num_classes)
        """

        # for teacher-forcing
        # images, labels are both lists of data - one element = one batch
        have_labels = False
        if len(inputs) == 2:    # Training
            have_labels = True
            images, labels = inputs
        else:                   # Test
            images = inputs

        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem3: input image into CNN and RNN sequentially.
        # NOTE: you can use teacher-forcing using labels or not
        # NOTE: you can modify below hint code

        hidden_state = torch.zeros(self.batch_size, self.rnn_hidden_size)
        cell_state = torch.zeros(self.batch_size, self.rnn_hidden_size)

        outputs = []

        if have_labels:
            # training code ...
            # teacher forcing by concatenating ()
            conv_outs = self.conv(images)
            if self.teacher_forcing:
                # output of conv is a list with dim B x [L, H]
                for label in labels:
                    output, _, _ = self.lstm(label, hidden_state, cell_state)
                    outputs.append(output)
            else:
                for conv_out in conv_outs:
                    output, _, _ = self.lstm(conv_out, hidden_state, cell_state)
                    outputs.append(output)

        else:
            # evaluation code ...
            conv_outs = self.conv(images)
            for conv_out in conv_outs:
                output, _, _ = self.lstm(conv_out, hidden_state, cell_state)
                outputs.append(output)

        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return outputs