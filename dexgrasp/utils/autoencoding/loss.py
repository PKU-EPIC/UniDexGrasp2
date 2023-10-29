import torch
import torch.nn as nn


class ChamferDistance(nn.Module):

    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, x, y):
        """
        The inputs are sets of d-dimensional points:
        x = {x_1, ..., x_n} and y = {y_1, ..., y_m}.

        Arguments:
            x: a float tensor with shape [b, d, n].
            y: a float tensor with shape [b, d, m].
        Returns:
            a float tensor with shape [].
        """
        x = x.unsqueeze(3)  # shape [b, d, n, 1]
        y = y.unsqueeze(2)  # shape [b, d, 1, m]

        # compute pairwise l2-squared distances
        d = torch.pow(x - y, 2)  # shape [b, d, n, m]
        d = d.sum(1)  # shape [b, n, m]

        min_for_each_x_i, _ = d.min(dim=2)  # shape [b, n]
        min_for_each_y_j, _ = d.min(dim=1)  # shape [b, m]

        distance = min_for_each_x_i.sum(1) + min_for_each_y_j.sum(1)  # shape [b]
        return distance.mean(0)
