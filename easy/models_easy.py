import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def weights_init(m):
    torch.nn.init.xavier_uniform(m.weight)

class CustomCNN(nn.Module):
    def __init__(self, cnn_input_size, cnn_hidden_size):
        # NOTE: you can freely add hyperparameters argument
        super(CustomCNN, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem1-1: define cnn model        
        # ResNet

        self.cnn_input_size = cnn_input_size
        self.cnn_hidden_size = cnn_hidden_size

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

        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, inputs):
        """
        For reference (shape example)
        inputs: Batch size X (Sequence_length, Channel=1, Height, Width)
        outputs: (Sequence_length X Batch_size, Hidden_dim)
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

        outputs = np.zeros(inputs.shape[0], self.cnn_hidden_size)

        for x in inputs[,:,,,]:
            x = x.squeeze()

            out = self.conv2(self.conv1(x))
            residual = out
            out = self.resblock1(out) + residual
            out = self.conv4(self.conv3(out))
            residual = out
            out = self.resblock2(out) + residual
            out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
            out = nn.Linear(out.shape[0] * out.shape[1] * out.shape[2] * out.shape[3], self.cnn_hidden_size)
            outputs = torch.cat([outputs, out], dim=0)

        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return outputs


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, vocab_size, num_layers=1, dropout=0.0):
        super(LSTM, self).__init__()

        # define the properties
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem2-1: Define lstm and input, output projection layer to fit dimension
        # output fully connected layer to project to the size of the class

        # you can either use torch LSTM or manually define it
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.fc_in = nn.Linear(vocab_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
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
                 cnn_hidden_size=256, rnn_hidden_size=512, rnn_num_layers=1, rnn_dropout=0, teacher_forcing=False):
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
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        self.conv = CustomCNN(self.cnn_input_dim, self.cnn_hidden_size)
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

        batch_size = inputs.shape[0]
        hidden_state = torch.zeros()
        cell_state = torch.zeros()

        if have_labels:
            # training code ...
            # teacher forcing by concatenating ()
            if self.teacher_forcing:
                conv_out = self.conv(images)
                outputs = self.lstm(labels)
            else:
                conv_out = self.conv(images)
                outputs = self.lstm(conv_out)

        else:
            # evaluation code ...
            conv_out = self.conv(images)
            outputs = self.lstm(conv_out)

        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return outputs