import torch
from torch import nn
import torch.nn.functional as F


class KeyNet(nn.Module):
    def __init__(self,
                 num_chans,
                 num_temporal_filts: int = 64,
                 num_spatial_filts: int = 2,
                 num_out_feats: int = 22):
        super(KeyNet, self).__init__()
        self.temporalizer = nn.Flatten(start_dim=0, end_dim=2)
        self.temporal_block = nn.Sequential(
            nn.Conv1d(1, num_temporal_filts, 9, stride=3),
            nn.BatchNorm1d(num_temporal_filts),
            nn.LeakyReLU(0.05),
            nn.Conv1d(num_temporal_filts, num_temporal_filts, 9, stride=3),
            nn.BatchNorm1d(num_temporal_filts),
            nn.LeakyReLU(0.05),
        )

        self.rnn = nn.LSTM(num_temporal_filts, num_temporal_filts, batch_first=True)

        num_spatiotemporal_filts = num_spatial_filts * num_temporal_filts
        self.spatial_block = nn.Sequential(
            nn.Conv3d(num_temporal_filts, num_spatiotemporal_filts, (1, 3, 3),
                      padding='same', groups=num_temporal_filts),
            nn.BatchNorm3d(num_spatiotemporal_filts),
            nn.LeakyReLU(0.05),
            nn.Conv3d(num_spatiotemporal_filts, num_out_feats, (1, 3, 3),
                      padding='same'),
            nn.BatchNorm3d(num_out_feats),
            nn.LeakyReLU(0.05),
            nn.Flatten(start_dim=3),
            nn.Linear(num_chans, 1)
        )

    def forward(self, x):
        batch_size, _, window_len, height, width = x.shape
        # Move channels into the sample dimension to process them in parallel
        temporal_x = self.temporalizer(x.permute(0, 3, 4, 1, 2))
        # Extract temporal features from each channel, returns (batch_size * num_chans, timesteps, num_temporal_filts)
        temporal_feats, _ = self.rnn(self.temporal_block(temporal_x).swapaxes(1, 2))
        print(temporal_feats.shape)
        # Move channels back into the channel dimension, returns (batch_size, h, w, num_temporal_filts, timesteps)
        temporal_feats = torch.unflatten(temporal_feats.swapaxes(1, 2), 0, torch.Size([batch_size, height, width]))
        print(temporal_feats.shape)
        # Extract spatiotemporal features, returns (batch_size, num_spatiotemporal_filts, timesteps, h, w)
        spatial_feats = self.spatial_block(temporal_feats.permute(0, 3, 4, 1, 2))
        return torch.squeeze(spatial_feats)


class SupConKeyNet(nn.Module):
    def __init__(self, encoder, head: str = 'linear', dim_in: int = 512, feat_dim: int = 128):
        super(SupConKeyNet, self).__init__()
        self.encoder = encoder
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(f"Head {head} not supported.")

    def forward(self, x):
        feat = F.normalize(self.encoder(x), dim=1)
        return F.normalize(self.head(feat), dim=1)


class LinearClassifier(nn.Module):
    def __init__(self, dim_in: int = 512, num_classes: int = 27):
        super(LinearClassifier, self).__init__()
        self.classifier = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.classifier(x)


if __name__ == "__main__":
    bs = 16
    in_chans = 1
    time = 240
    h = 4
    w = 8
    test_data = torch.ones((bs, in_chans, time, h, w))

    num_chans = 32
    num_keys = 4
    num_fingers = 3
    model = KeyNet(num_chans)
    out = model(test_data)
    print(out.shape)
