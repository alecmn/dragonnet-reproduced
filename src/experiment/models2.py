# Setup
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import math


def binary_classification_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    t_pred = (t_pred + 0.001) / 1.002
    loss = torch.sum(F.binary_cross_entropy_with_logits(t_true, t_pred))
    # print(f"T_true: {t_true}")
    # print(f"T_pred: {t_pred}")
    # print(F.binary_cross_entropy(t_true, t_pred))
    return loss


def regression_loss(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    loss0 = torch.sum((1. - t_true) * torch.square(y_true - y0_pred))
    loss1 = torch.sum(t_true * torch.square(y_true - y1_pred))

    return loss0 + loss1


def ned_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]

    t_pred = concat_pred[:, 1]
    return torch.sum(F.binary_cross_entropy_with_logits(t_true, t_pred))


def dead_loss(concat_true, concat_pred):
    return regression_loss(concat_true, concat_pred)


def dragonnet_loss_binarycross(concat_pred, concat_true):
    # print(regression_loss(concat_true, concat_pred))
    # print(binary_classification_loss(concat_true, concat_pred))
    return regression_loss(concat_true, concat_pred) + binary_classification_loss(concat_true, concat_pred)


# find out why this is neccessary
# not dicussed
class EpsilonLayer(nn.Module):
    def __init__(self):
        super().__init__()

        # building epsilon trainable weight
        self.weights = nn.Parameter(torch.Tensor(1, 1))

        # initializing weight parameter with RandomNormal
        nn.init.normal_(self.weights)

    def forward(self, inputs):
        return torch.mm(torch.ones_like(inputs)[:, 0:1], self.weights.T)


def make_tarreg_loss(ratio=1., dragonnet_loss=dragonnet_loss_binarycross):
    def tarreg_ATE_unbounded_domain_loss(concat_pred, concat_true):
        vanilla_loss = dragonnet_loss(concat_pred, concat_true)

        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]
        t_pred = concat_pred[:, 2]

        epsilons = concat_pred[:, 3]
        t_pred = (t_pred + 0.01) / 1.02
        # t_pred = tf.clip_by_value(t_pred,0.01, 0.99,name='t_pred')

        y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

        h = t_true / t_pred - (1 - t_true) / (1 - t_pred)

        y_pert = y_pred + epsilons * h
        targeted_regularization = torch.sum(torch.square(y_true - y_pert))

        # final
        loss = vanilla_loss + ratio * targeted_regularization
        return loss

    return tarreg_ATE_unbounded_domain_loss


# weight initialzation function
def weights_init(params):
    if isinstance(params, nn.Linear):
        torch.nn.init.normal_(params.weight, mean=0.0, std=1.0)
        torch.nn.init.zeros_(params.bias)


class DragonNet(nn.Module):
    """
    3-headed dragonet architecture

    Args:
        in_channels: number of features of the input image ("depth of image")
        hidden_channels: number of hidden features ("depth of convolved images")
        out_features: number of features in output layer
    """

    def __init__(self, in_features, out_features=[200, 100, 1]):
        super(DragonNet, self).__init__()

        # representation layers 3 : block1
        # units in kera = out_features

        self.representation_block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features[0]),
            nn.ELU(),
            nn.Linear(in_features=out_features[0], out_features=out_features[0]),
            nn.ELU(),
            nn.Linear(in_features=out_features[0], out_features=out_features[0]),
            nn.ELU()
        )

        # -----------Propensity Head
        self.t_predictions = nn.Sequential(nn.Linear(in_features=out_features[0], out_features=out_features[2]),
                                           nn.Sigmoid())

        # -----------t0 Head
        self.t0_head = nn.Sequential(nn.Linear(in_features=out_features[0], out_features=out_features[1]),
                                     nn.ELU(),
                                     nn.Linear(in_features=out_features[1], out_features=out_features[1]),
                                     nn.ELU(),
                                     nn.Linear(in_features=out_features[1], out_features=out_features[2])
                                     )

        # ----------t1 Head
        self.t1_head = nn.Sequential(nn.Linear(in_features=out_features[0], out_features=out_features[1]),
                                     nn.ELU(),
                                     nn.Linear(in_features=out_features[1], out_features=out_features[1]),
                                     nn.ELU(),
                                     nn.Linear(in_features=out_features[1], out_features=out_features[2])
                                     )

        self.epsilon = EpsilonLayer()

    def init_params(self, std=1):
        """
        Initialize layer parameters. Sample weight from Gaussian distribution
        and bias uniform distribution.

        Args:
            std: Standard deviation of Random normal distribution (default: 1)
        """
        self.representation_block.apply(weights_init)
        # self.t_predictions.apply(weights_init)
        # self.t0_head.apply(weights_init)
        # self.t1_head.apply(weights_init)

    def forward(self, x):
        # print(x)
        x = self.representation_block(x)
        # print(f"Repr block: {x}")

        # ------propensity scores
        propensity_head = self.t_predictions(x)
        epsilons = self.epsilon(propensity_head)
        # print(f"Prop head: {propensity_head}")

        # ------t0
        t0_out = self.t0_head(x)
        # print(f"t0 out: {t0_out}")

        # ------t1
        t1_out = self.t1_head(x)
        # print(f"t1_out: {t1_out}")

        # print(t0_out, t1_out, propensity_head, epsilons)
        return torch.cat((t0_out, t1_out, propensity_head, epsilons), 1)


class TarNet(nn.Module):
    """
    3-headed tarnet architecture

    Args:
        in_channels: number of features of the input image ("depth of image")
        hidden_channels: number of hidden features ("depth of convolved images")
        out_features: number of features in output layer
    """

    def __init__(self, in_features, out_features=[200, 100, 1]):
        super(TarNet, self).__init__()

        # representation layers 3 : block1
        # units in kera = out_features

        self.representation_block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features[0]),
            nn.ELU(),
            nn.Linear(in_features=out_features[0], out_features=out_features[0]),
            nn.ELU(),
            nn.Linear(in_features=out_features[0], out_features=out_features[0]),
            nn.ELU()
        )

        # -----------Propensity Head
        self.t_predictions = nn.Sequential(nn.Linear(in_features=out_features[0], out_features=out_features[2]),
                                           nn.Sigmoid())

        # -----------t0 Head
        self.t0_head = nn.Sequential(nn.Linear(in_features=out_features[0], out_features=out_features[1]),
                                     nn.ELU(),
                                     nn.Linear(in_features=out_features[1], out_features=out_features[1]),
                                     nn.ELU(),
                                     nn.Linear(in_features=out_features[1], out_features=out_features[2])
                                     )

        # ----------t1 Head
        self.t1_head = nn.Sequential(nn.Linear(in_features=out_features[0], out_features=out_features[1]),
                                     nn.ELU(),
                                     nn.Linear(in_features=out_features[1], out_features=out_features[1]),
                                     nn.ELU(),
                                     nn.Linear(in_features=out_features[1], out_features=out_features[2])
                                     )

        self.epsilon = EpsilonLayer()

    def init_params(self, std=1):
        """
        Initialize layer parameters. Sample weight from Gaussian distribution
        and bias uniform distribution.

        Args:
            std: Standard deviation of Random normal distribution (default: 1)
        """
        self.representation_block.apply(weights_init)
        # self.t_predictions.apply(weights_init)
        # self.t0_head.apply(weights_init)
        # self.t1_head.apply(weights_init)

    def forward(self, x):
        # print(x)
        rep_block = self.representation_block(x)
        # print(f"Repr block: {x}")

        # ------propensity scores
        propensity_head = self.t_predictions(x)
        epsilons = self.epsilon(propensity_head)
        # print(f"Prop head: {propensity_head}")

        # ------t0
        t0_out = self.t0_head(rep_block)
        # print(f"t0 out: {t0_out}")

        # ------t1
        t1_out = self.t1_head(rep_block)
        # print(f"t1_out: {t1_out}")

        # print(t0_out, t1_out, propensity_head, epsilons)
        return torch.cat((t0_out, t1_out, propensity_head, epsilons), 1)
