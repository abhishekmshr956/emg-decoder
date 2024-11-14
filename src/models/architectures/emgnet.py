import torch
from torch import nn

from emg_decoder.src.utils import load_config


class PrintLayer(nn.Module):
    def __init__(self, label=None):
        super(PrintLayer, self).__init__()
        self.label = label

    def forward(self, x):
        # Do your print / debug stuff here
        print(self.label, x.shape)
        return x


class EMGNet(nn.Module):
    def __init__(self,
                 num_temporal_filts: int = 8,
                 num_spatial_filts: int = 2,
                 num_chans: int = 64,
                 window_length: int = 100,
                 num_classes: int = 6,
                 embed_dim: int = 64,
                 num_layers: int = 1,
                 effective_fs: float = 2000.0,
                 p_dropout: float = 0.5,
                 avgpool_factor: int = 4,
                 rnn_type: str = 'lstm') -> None:
        """
        :param num_temporal_filts: Number of temporal filters in the first convolutional layer
        :param num_spatial_filts: Number of spatial filters in the second convolutional layer
        :param num_chans: Number of channels in the input data
        :param p_dropout: Probability of dropout
        :param window_length: Length of the window in samples
        :param num_classes: Number of classes (keys)
        :param embed_dim: Embedding dimension for the GRU
        :param num_layers: Number of layers for the GRU
        :param effective_fs: Effective sampling frequency of the data (in Hz)
        :param avgpool_factor: Factor for the first average pooling layer
        :param rnn_type: Type of RNN to use (LSTM or GRU)
        """
        super(EMGNet, self).__init__()

        self.F1 = num_temporal_filts
        self.D = num_spatial_filts
        self.C = num_chans
        self.F2 = self.D * self.F1

        self.p = p_dropout
        self.T = window_length
        self.num_classes = num_classes
        self.avgpool_factor = avgpool_factor
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, int(0.1 * effective_fs) + 1), padding='same'),
            nn.Conv2d(self.F1, self.D * self.F1, (self.C, 1), groups=self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, self.avgpool_factor)),
            nn.Dropout(self.p)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(self.F2,
                      self.F2,
                      (1, 2 * (self.T // (self.avgpool_factor * 2)) + 1),
                      groups=self.F2,
                      padding='same'),
            nn.Conv2d(self.F2, self.F2, (1, 1)),
            nn.ELU(),
            nn.Flatten(start_dim=2, end_dim=-1),
            nn.Linear(self.T // self.avgpool_factor, 1),
            nn.Dropout(self.p),
        )

        if rnn_type.lower() == 'lstm':
            self.rnn = nn.Sequential(
                nn.LSTM(self.D * self.F1, self.embed_dim, num_layers=self.num_layers, batch_first=True),
            )
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.Sequential(
                nn.GRU(self.D * self.F1, self.embed_dim, num_layers=self.num_layers, batch_first=True),
            )
        else:
            raise NotImplementedError(f"RNN type {rnn_type} not implemented.")

        self.classifier = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim),
                                        nn.LeakyReLU(0.05),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(self.embed_dim, self.num_classes),
                                        nn.LogSoftmax(dim=2))

    def forward(self, x):
        print("in forward of emgnet")
        batch_size, num_windows, num_channels, window_len = x.shape
        block1 = self.block1(x.view(-1, 1, num_channels, window_len))
        out = self.block2(block1)
        rnn_out, _ = self.rnn(out.view(batch_size, num_windows, -1))
        return self.classifier(rnn_out)


if __name__ == "__main__":
    config_path = "/configs/examples/emgnet_example.yaml"
    config = load_config(config_path)
    effective_sampling_freq = config['metadata']['sampling_frequency'] // config['preprocessing']['downsample_factor']
    # These values in seconds
    window_stride = int(0.50 * effective_sampling_freq)
    window_len = int(0.050 * effective_sampling_freq)
    interval_len = int(2.500 * effective_sampling_freq)
    # num_windows_per_seq = int(interval_len / window_stride)
    # print(window_stride, window_len, interval_len, num_windows_per_seq)

    batch_size = 4
    num_channels = 64
    num_windows_per_seq = 50
    # window_len = 100
    num_classes = 6

    net = EMGNet(num_temporal_filts=64,
                 num_spatial_filts=8,
                 num_chans=num_channels,
                 window_length=window_len,
                 num_classes=num_classes,
                 avgpool_factor=2)

    test_data = torch.rand(batch_size, num_windows_per_seq, num_channels, window_len)
    print("input shape", test_data.shape)
    print("output shape", net(test_data).shape)
