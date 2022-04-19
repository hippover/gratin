import torch
import numpy as np
from sklearn.metrics import f1_score
from ..data.datamodule import EMPTY_FIELD_VALUE
import torch.nn as nn
import torch

## Losses


def L2_loss(out, target, w):
    # w : whether each sample has to be included
    # for instance, OU don't have to be included in alpha,
    # while LW don't have to be included in theta
    mean_dims = torch.mean(torch.pow(out[w] - target[w], 2), dim=1)
    # concerned = torch.masked_select(mean_dims,w)
    return torch.mean(mean_dims)


def Category_loss(out, target, w):
    # w : whether each sample has to be included
    # return nn.CrossEntropyLoss()(torch.index_select(out,0,w), torch.index_select(target,0,w))
    l = nn.CrossEntropyLoss()(
        out[w],
        target[w].view(
            -1,
        ),
    )
    return l


## Metrics for training
def is_concerned(target):
    """Returns 1 if the sample should be included in the loss (based on the value of the target)

    Args:
        target (_type_): _description_

    Returns:
        _torch.Tensor_: tensor of booleans
    """
    if len(target.shape) == 1:
        return ~torch.isnan(target)
    else:
        return torch.eq(torch.sum(1 * torch.isnan(target), dim=1), 0)
    # return torch.eq(torch.sum(1 * torch.eq(target, EMPTY_FIELD_VALUE), dim=1), 0)
