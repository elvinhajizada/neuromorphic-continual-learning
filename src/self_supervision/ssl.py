import lightly
from lightly.models.modules import BarlowTwinsProjectionHead

import torch.distributed as dist
from torch import nn
import torch


class BarlowTwins(nn.Module):
    def __init__(self, backbone, head_size=(512, 1024, 2048)):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(*head_size)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z


class ModifiedBarlowTwinsLoss(torch.nn.Module):
    """Implementation of the Barlow Twins Loss from Barlow Twins[0] paper.
    This code specifically implements the Figure Algorithm 1 from [0].

    [0] Zbontar,J. et.al, 2021, Barlow Twins... https://arxiv.org/abs/2103.03230

    """

    def __init__(
            self,
            lambda_param: float = 5e-3,
            theta=20,
            gather_distributed: bool = False
    ):
        """Lambda param configuration with default value like in [0]

        Args:
            lambda_param:
                Parameter for importance of redundancy reduction term.
                Defaults to 5e-3 [0].
            gather_distributed:
                If True then the cross-correlation matrices from all gpus are
                gathered and summed before the loss calculation.
        """
        super(ModifiedBarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.gather_distributed = gather_distributed
        self.theta = theta

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:

        device = z_a.device

        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0)  # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0)  # NxD

        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N  # DxD

        # sum cross-correlation matrix between multiple gpus
        if self.gather_distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                c = c / world_size
                dist.all_reduce(c)

        # loss
        c_diff = (c - torch.eye(D, device=device)).pow(2)  # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        bt_loss = c_diff.sum()

        # norm_loss = 0
        # norm_loss=norm_loss.to(device)
        # if self.theta != 0:
        z = torch.cat((z_a_norm, z_b_norm), 0)

        norms = torch.norm(z, dim=1)

        nomrs_std = torch.std(norms)

        # print(nomrs_std)
        norm_loss = self.theta * nomrs_std

        loss = bt_loss + norm_loss

        return (loss, bt_loss, nomrs_std)