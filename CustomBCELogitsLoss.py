from torch import nn, Tensor
from typing import Iterable, Dict, List
from math import floor


class CustomBCELogitsLoss(nn.Module):
    """ """

    def __init__(
        self,
        loss_fct=nn.BCEWithLogitsLoss(),
    ):
        super(CustomBCELogitsLoss, self).__init__()
        self.loss_fct = loss_fct

    def forward(self, input: Tensor, target: Tensor, sources: List[str]):
        for idx, (inp, source) in enumerate(zip(input, sources)):
            if source in ["semeval", "misc"] and inp != 1:
                input[idx] = 0.5 * floor(inp * 3)
        return self.loss_fct(input, target)
