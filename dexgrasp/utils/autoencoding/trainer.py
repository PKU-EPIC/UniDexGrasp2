import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import Autoencoder, AutoencoderTransPN, AutoencoderPN
from loss import ChamferDistance


class Trainer:

    def __init__(self, num_steps, device):

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

        network = AutoencoderTransPN(k=128, num_points=1024)
        self.network = network.apply(weights_init).to(device)

        self.loss = ChamferDistance()
        self.optimizer = optim.Adam(self.network.parameters(), lr=5e-4, weight_decay=1e-5)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_steps, eta_min=1e-7)

    def train_step(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, num_points].
        Returns:
            a float number.
        """

        _, x_restored = self.network(x)
        loss = self.loss(x, x_restored)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint)

    def evaluate(self, x):

        with torch.no_grad():
            _, x_restored = self.network(x)
            loss = self.loss(x, x_restored)

        return loss.item(), x_restored, _