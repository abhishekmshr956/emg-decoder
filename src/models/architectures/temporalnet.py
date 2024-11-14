from typing import Tuple

import torch
from omegaconf import OmegaConf
from torch import nn


class ConvGroup(nn.Module):
    def __init__(self,
                 conv_layers: list,
                 use_batchnorm: bool = True,
                 use_maxpool: bool = True) -> None:

        """
        Module to create a group of Conv2d layers, each followed by batchnorm, maxpool, and LeakyReLU activations.
        :param conv_layers: List of 5-tuples (out_feats, kernel, stride, pool_kernel, pool_stride).
        :param use_batchnorm: Whether to add batchnorm to each layer.
        :param use_maxpool: Whether to add maxpool to each layer.
        """
        super().__init__()
        layers = []
        in_feats = 1
        for (out_feats, kernel, stride, pool_kernel, pool_stride) in conv_layers:
            layers.append(
                nn.Conv2d(in_feats, out_feats, kernel, stride=stride)
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_feats))
            if use_maxpool:
                layers.append(nn.MaxPool2d(pool_kernel, stride=pool_stride))
            layers.append(nn.LeakyReLU(0.05))

            in_feats = out_feats

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class TemporalClassifier(nn.Module):
    def __init__(self,
                 conv_layers: list[Tuple],
                 num_chans: int = 32,
                 num_classes: int = 6,
                 rnn_type: str = 'lstm') -> None:
        """
        :param conv_layers: List of 5-tuples (out_feats, kernel, stride, pool_kernel, pool_stride).
        :param num_chans: Number of EMG input channels.
        :param num_classes: Number of output classes (number of keys + 1 for blank symbol).
        """
        super().__init__()
        self.conv_group = ConvGroup(conv_layers)
        self.flattener = nn.Flatten(1, 2)
        self.linear = nn.Sequential(nn.Linear(num_chans * conv_layers[-1][0], 128),
                                    nn.LeakyReLU(0.05),
                                    nn.Dropout(p=0.5))

        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(128, 64, num_layers=2, batch_first=True)
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(128, 64, num_layers=2, batch_first=True)
        else:
            raise NotImplementedError(f"RNN type {rnn_type} not implemented.")

        self.classifier = nn.Sequential(nn.Linear(64, 128),
                                        nn.LeakyReLU(0.05),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(128, 64),
                                        nn.LeakyReLU(0.05),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(64, num_classes),
                                        nn.LogSoftmax(dim=2))

    def forward(self, x):
        independent_chan_feats = self.conv_group(torch.unsqueeze(x, dim=1))
        flat = self.flattener(independent_chan_feats)
        mixed_chan_feats = self.linear(torch.swapaxes(flat, 1, 2))
        rnn_out, _ = self.rnn(mixed_chan_feats)
        return self.classifier(rnn_out)


if __name__ == "__main__":
    # Since we want to keep channels separate, kernel should be 1 on channel dimension
    # data should be of shape (batch_size, num_emg_chans, num_timesteps, num_feats=1)
    # (out_feats, kernel, stride, pool_kernel, pool_stride)
    conv_layers = [(4, (1, 15), 1, (1, 3), (1, 3)),
                   (8, (1, 15), 1, (1, 3), (1, 3)),
                   (16, (1, 15), 1, (1, 3), 1),
                   (32, (1, 15), 1, (1, 3), 1),
                   (32, (1, 15), 1, (1, 3), 1),
                   (50, (1, 15), 1, (1, 3), 1)]

    path = "/Users/johnzhou/research/emg_decoder/configs/temporal_lstm_shortinput/nevin_28apr_1715_w05_s02.yaml"
    # conv_layers = OmegaConf.to_container(OmegaConf.load(path), resolve=True)['model']['conv_layers']
    # for layer in conv_layers:
    #     print(layer)
    batch_size = 4
    num_channels = 64
    num_sec = 2.5
    downsample_factor = 8
    fs = 4000
    num_timesteps = int(num_sec * fs / downsample_factor)
    bellnet = TemporalClassifier(conv_layers, num_chans=num_channels)
    test_data = torch.rand(batch_size, num_channels, num_timesteps)
    print(test_data.shape)
    print(bellnet(test_data).shape)

