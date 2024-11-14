import editdistance
import numpy as np
import torch
from torch import tensor, nn

from emg_decoder.src.data.utils import remove_adj_duplicates


class LevenshteinLoss(nn.Module):
    def __init__(self):
        super(LevenshteinLoss, self).__init__()

    @staticmethod
    def forward(y_softmax: tensor, y: tensor, target_lengths: tensor) -> float:
        """
        Calculates the average Levenshtein distance between the predicted and true labels for a batch of data.
        :param y_softmax: Softmax output of model for a batch of data.
        :param y: Ground truth labels for a batch of data.
        :param target_lengths: Lengths of ground truth labels for a batch of data.
        :return: Percentage accuracy for batch of data as average Levenshtein distance divided by length of true sequence.
        """
        total_samples = 0
        percentages = 0
        current_idx = 0
        preds = torch.argmax(y_softmax, dim=2, keepdim=False).T
        preds = preds.tolist()
        decoded = [remove_adj_duplicates(np.array(pred)) for pred in preds]

        for idx, pred in enumerate(decoded):
            pred = ''.join(str(x) for x in list(pred))
            true = y[current_idx: current_idx + target_lengths[idx]].long().tolist()
            true = ''.join(str(x) for x in true)
            current_idx += target_lengths[idx]
            distance = editdistance.eval(pred, true)
            total_samples += 1
            if target_lengths[idx] == 0:
                if distance == 0:
                    percentages += 1
                else:
                    percentages += 0
            else:
                percentages += distance / target_lengths[idx]

        return float(1 - percentages / total_samples)
