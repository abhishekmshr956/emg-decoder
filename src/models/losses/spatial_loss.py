import numpy as np

import torch
from torch import nn, tensor
import torch.nn.functional as F

SPECIAL_CODES = {65362: 'up', 65293: 'enter', 65507: 'ctrl', 32: 'space', 65289: 'tab', 65288: 'backspace',
                 65505: 'shift', 65513: 'alt'}
SPECIAL_KEYS = dict((code, key) for key, code in SPECIAL_CODES.items())


class Key:
    def __init__(self, code: int):
        self.code = code
        self.coordinates = None

    def set_coordinates(self, x, y):
        self.coordinates = (x, y)


class Keyboard:
    layout = [['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
              ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
              ['z', 'x', 'c', 'v', 'b', 'n', 'm'],
              [' '],
              ['#']]  # '#' is a placeholder for blank key
    flat_char_layout = [key for row in layout for key in row]
    flat_ord_layout = [ord(key) for key in flat_char_layout]
    horiz_spacing = 1
    vert_spacing = 1
    row_offsets = [0, 0.2, 0.4, 4.5, 4.5]

    def __init__(self):
        self.keys = self.set_keys()
        self.coord_array = self.get_coord_array()
        self.ascii_sorted_coords = self.coord_array[np.argsort(self.flat_ord_layout)]

    def set_keys(self):
        keys = []
        for row_idx in range(len(self.layout)):
            keys.append([])
            row = self.layout[row_idx]
            for col_idx in range(len(row)):
                key = Key(ord(row[col_idx]))
                key.set_coordinates(self.horiz_spacing * col_idx + self.row_offsets[row_idx],
                                    self.vert_spacing * row_idx)
                keys[row_idx].append(key)
        return keys

    def get_coord_array(self):
        return np.array([key.coordinates for row in self.keys for key in row])


class SpatialLoss(nn.Module):
    def __init__(self, coordinates: tensor):
        """
        Spatial loss function.
        :param coordinates: A 2D matrix of shape (num_classes, num_coords) where coordinates[i] are the Euclidean
        coordinates corresponding to class i.
        """
        super(SpatialLoss, self).__init__()
        self.coordinates = torch.unsqueeze(coordinates, 1)

    def forward(self, raw_logits: tensor, target: tensor):
        bsz = raw_logits.shape[0]
        softmax = F.softmax(raw_logits, dim=1)
        target_coords = torch.index_select(self.coordinates, 0, target)  # this is not differentiable w.r.t. index
        distances = self.coordinates.expand(-1, bsz, -1) - target_coords.swapaxes(0, 1)
        l2_sq = torch.sum(distances ** 2, dim=2).T
        return torch.sum(softmax * l2_sq)


if __name__ == '__main__':
    keyboard = Keyboard()
    loss = SpatialLoss(tensor(keyboard.ascii_sorted_coords))
    bsz = 4
    n_classes = 27
    logits = torch.randn((bsz, n_classes))
    target = torch.randint(0, n_classes, (bsz,))
    print(loss(logits, target))
