import torch
from torch import nn
import numpy as np


class PrintLayer(nn.Module):
    def __init__(self, label=None):
        super(PrintLayer, self).__init__()
        self.label = label

    def forward(self, x):
        # Do your print / debug stuff here
        print(self.label, x.shape)
        return x


class CC(nn.Module):
    def __init__(self,
                 num_temporal_filts: int = 8,
                 num_spatial_filts: int = 2,
                 num_chans: int = 32,
                 window_length: int = 100,
                 p_dropout: float = 0.5,
                 avgpool_factor: int = 4) -> None:
        """
        :param num_temporal_filts: Number of temporal filters in the first convolutional layer
        :param num_spatial_filts: Number of spatial filters in the second convolutional layer
        :param num_chans: Number of channels in the input data
        :param p_dropout: Probability of dropout
        :param window_length: Length of the window in samples
        :param avgpool_factor: Factor for the first average pooling layer
        """
        super(CC, self).__init__()
        self.F1 = num_temporal_filts
        self.D = num_spatial_filts
        self.C = num_chans
        self.F2 = self.D * self.F1

        self.p = p_dropout
        self.T = window_length
        self.avgpool_factor1 = avgpool_factor

        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, 9), padding='same'),
            nn.Conv2d(self.F1, self.D * self.F1, (self.C, 1), groups=self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, self.avgpool_factor1)),
            nn.Dropout(self.p)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(self.F2,
                      self.F2,
                      (1, 2 * (self.T // (self.avgpool_factor1 * 2)) + 1),
                      groups=self.F2,
                      padding='same'),
            nn.Conv2d(self.F2, self.F2, (1, 1)),
            nn.ELU(),
            nn.Flatten(start_dim=2, end_dim=-1),
        )
        #batches X features X timesteps/2

        #change this lol
        self.block3 = nn.Sequential(
            nn.Linear(6, 6),
            nn.ReLU()
        )

        self.embed_dim = 64
        self.num_layers = 1

        self.rnn = nn.Sequential(
            nn.LSTM(self.D * self.F1, self.embed_dim, num_layers=self.num_layers, batch_first=True),
        )

        self.num_classes = 6

        self.classifier = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim),
                                        nn.LeakyReLU(0.05),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(self.embed_dim, self.num_classes),
                                        nn.LogSoftmax(dim=2))

    def forward(self, x, angles):
        x = x.swapaxes(1, 2).unsqueeze(1)
        block1 = self.block1(x)
        # print("block1", block1.shape)
        # print("block2", self.block2(block1).shape)
        block2 = self.block2(block1)
        block2 = block2.swapaxes(1, 2) #batches x timesteps/2 x features

        angles = self.block3(angles)

        #want angles to be of form (batches, timesteps/avgpool_factor1, 6)
        block2 = np.concatenate((block2, angles), axis=2)

        rnn_out, _ = self.rnn(block2)
        return self.classifier(rnn_out)
