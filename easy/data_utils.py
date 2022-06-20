import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

class CustomCNN(nn.Module):
    def __init__(self, cnn_input_size, cnn_hidden_size, vocab_size):
        # NOTE: you can freely add hyperparameters argument
        super(CustomCNN, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem1-1: define cnn model
        # VGG16

        self.cnn_input_size = cnn_input_size
        self.cnn_hidden_size = cnn_hidden_size
        self.vocab_size = vocab_size

        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(True)

        self.fc = nn.Linear(512 * 7 * 7, self.vocab_size)

        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, inputs):
        """
        For reference (shape example)
        inputs: [Batch size x Sequence_length, Channel=1, Height, Width]
        outputs: [Sequence_length X Batch_size, Hidden_dim]
        """
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem1-2: code CNN forward path

        inputs = inputs.cuda()

        out = self.relu(self.conv1_2(self.relu(self.conv1_1(inputs))))

        out = self.relu(self.conv2_2(self.relu(self.conv2_1(out))))
        out = self.maxpool(out)

        out = self.relu(self.conv3_3(self.relu(self.conv3_2(self.relu(self.conv3_1(out))))))

        out = self.relu(self.conv4_3(self.relu(self.conv4_2(self.relu(self.conv4_1(out))))))

        out = self.relu(self.conv5_3(self.relu(self.conv5_2(self.relu(self.conv5_1(out))))))
        out = self.maxpool(out) # [256 * 5, 512, 7, 7]

        out = out.view(-1, out.size(1) * out.size(2) * out.size(3)) # [256 * 5, 512 * 7 * 7]
        outputs = self.relu(self.fc(out))  # outputs = [BS, Hidden]

        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return outputs

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


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

        output, hidden = self.lstm(feature, (h, c))
        output = self.fc_out(output)
        output = output[-1, :, :].clone() + feature
        h_next, c_next = hidden

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
        self.conv = CustomCNN(self.cnn_input_dim, self.cnn_hidden_size, self.num_classes)
        self.lstm = LSTM(self.rnn_input_dim, self.rnn_hidden_size, self.num_classes, self.rnn_num_layers, self.rnn_dropout)
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

        hidden_state = torch.zeros(self.rnn_num_layers, images.size(0), self.rnn_hidden_size).cuda()
        cell_state = torch.zeros(self.rnn_num_layers, images.size(0), self.rnn_hidden_size).cuda()

        img = images.view(images.size(0) * self.sequence_length, 1, 28, 28) # [BxS, C, H, W]

        if have_labels:
            # training code ...
            # teacher forcing by concatenating ()
            labels = labels.transpose(0, 1)  # [S, B, 26]

            conv_out = self.conv(img)
            conv_outs = conv_out.view(self.sequence_length, images.size(0), -1)  # [S, B, 26]
            if self.teacher_forcing:
                outputs, _, _ = self.lstm(conv_outs + labels, hidden_state, cell_state)
            else:
                outputs, _, _ = self.lstm(conv_outs, hidden_state, cell_state)

        else:
            # evaluation code ...
            conv_out = self.conv(img)
            conv_outs = conv_out.view(self.sequence_length, images.size(0), -1)  # [S, B, 26]
            outputs, _, _ = self.lstm(conv_outs, hidden_state, cell_state)

        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return outputs

