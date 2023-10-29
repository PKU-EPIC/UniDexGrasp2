import torch.nn as nn
from typing import List, Optional, Tuple
from maniskill_learn.networks.backbones.pointnet import getPointNetWithInstanceInfoDex, getPointNetWithInstanceInfo, getSparseUnetWithInstanceInfo
import torch


class Autoencoder(nn.Module):

    def __init__(self, k, num_points):
        """
        Arguments:
            k: an integer, dimension of the representation vector.
            num_points: an integer.
        """
        super(Autoencoder, self).__init__()

        # ENCODER

        pointwise_layers = []
        num_units = [3, 64, 128, 128, 256, k]

        for n, m in zip(num_units[:-1], num_units[1:]):
            pointwise_layers.extend([
                nn.Conv1d(n, m, kernel_size=1, bias=False),
                nn.BatchNorm1d(m),
                nn.ReLU(inplace=True)
            ])

        self.pointwise_layers = nn.Sequential(*pointwise_layers)
        self.pooling = nn.AdaptiveMaxPool1d(1)

        # DECODER

        self.decoder = nn.Sequential(
            nn.Conv1d(k, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, num_points * 3, kernel_size=1)
        )

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, num_points].
        Returns:
            encoding: a float tensor with shape [b, k].
            restoration: a float tensor with shape [b, 3, num_points].
        """

        b, _, num_points = x.size()
        x = self.pointwise_layers(x)  # shape [b, k, num_points]
        encoding = self.pooling(x)  # shape [b, k, 1]

        x = self.decoder(encoding)  # shape [b, num_points * 3, 1]
        restoration = x.view(b, 3, num_points)

        return encoding, restoration

class AutoencoderPN(nn.Module):

    def __init__(self, k, num_points):
        """
        Arguments:
            k: an integer, dimension of the representation vector.
            num_points: an integer.
        """
        super(AutoencoderPN, self).__init__()

        # ENCODER

        self.backbone = PointNetBackbone(pc_dim=5, feature_dim=128)
        # pointwise_layers = []
        # num_units = [3, 64, 128, 128, 256, k]

        # for n, m in zip(num_units[:-1], num_units[1:]):
        #     pointwise_layers.extend([
        #         nn.Conv1d(n, m, kernel_size=1, bias=False),
        #         nn.BatchNorm1d(m),
        #         nn.ReLU(inplace=True)
        #     ])

        # self.pointwise_layers = nn.Sequential(*pointwise_layers)
        # self.pooling = nn.AdaptiveMaxPool1d(1)

        # DECODER

        self.decoder = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, num_points * 3, kernel_size=1)
        )

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, num_points].
        Returns:
            encoding: a float tensor with shape [b, k].
            restoration: a float tensor with shape [b, 3, num_points].
        """

        b, _, num_points = x.size()
        
        data = torch.cat((x, torch.ons((x.shape[0],x.shape[1], 2), device = x.device)), dim = 2)
        encoding = self.backbone(data)[0].unsqueeze(2)  # shape [b, k, 1]

        x = self.decoder(encoding)  # shape [b, num_points * 3, 1]
        restoration = x.view(b, 3, num_points)

        return encoding, restoration

class AutoencoderTransPN(nn.Module):

    def __init__(self, k, num_points):
        """
        Arguments:
            k: an integer, dimension of the representation vector.
            num_points: an integer.
        """
        super(AutoencoderTransPN, self).__init__()

        # ENCODER
        self.backbone = TransPointNetBackbone(pc_dim=3, feature_dim=128)

        # DECODER

        self.decoder = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, num_points * 3, kernel_size=1)
        )

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, num_points].
        Returns:
            encoding: a float tensor with shape [b, k].
            restoration: a float tensor with shape [b, 3, num_points].
        """

        b, _, num_points = x.size()
        data = {"pc": x, "state": torch.ones((b,128), device = x.device), "mask": torch.ones((b,num_points, 2), device = x.device)}
        encoding = self.backbone(data)[0].unsqueeze(2)  # shape [b, k, 1]
        x = self.decoder(encoding)  # shape [b, num_points * 3, 1]
        restoration = x.view(b, 3, num_points)

        return encoding, restoration

class PointNetBackbone(nn.Module):
    def __init__(
        self,
        pc_dim: int,
        feature_dim: int,
        pretrained_model_path: Optional[str] = None,
    ):
        super().__init__()
        # self.save_hyperparameters()
        self.pc_dim = pc_dim
        self.feature_dim = feature_dim
        self.backbone = getPointNet({
                'input_feature_dim': self.pc_dim,
                'feat_dim': self.feature_dim
            })

        if pretrained_model_path is not None:
            print("Loading pretrained model from:", pretrained_model_path)
            state_dict = torch.load(
                pretrained_model_path, map_location="cpu"
            )["state_dict"]
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=False,
            )
            if len(missing_keys) > 0:
                print("missing_keys:", missing_keys)
            if len(unexpected_keys) > 0:
                print("unexpected_keys:", unexpected_keys)
            
    
    def forward(self, input_pc):
        others = {}
        return self.backbone(input_pc), others

class TransPointNetBackbone(nn.Module):
    def __init__(
        self,
        pc_dim: int = 6,
        feature_dim: int = 128,
        state_dim: int = 128,
        use_seg: bool = True,
    ):
        super().__init__()

        cfg = {}
        cfg["state_dim"] = 128
        cfg["feature_dim"] = feature_dim
        cfg["pc_dim"] = pc_dim
        cfg["output_dim"] = feature_dim
        if use_seg:
            cfg["mask_dim"] = 2
        else:
            cfg["mask_dim"] = 0

        self.transpn = getPointNetWithInstanceInfo(cfg)

    def forward(self, input_pc):
        others = {}
        # input_pc["pc"] = torch.cat([input_pc["pc"], torch.ones((input_pc["pc"].shape[0],input_pc["pc"].shape[1],2), device = input_pc["pc"].device)], dim = -1)
        input_pc["pc"] = torch.cat([input_pc["pc"].permute(0, 2, 1), input_pc["mask"]], dim = -1)
        return self.transpn(input_pc), others
