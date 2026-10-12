"""Loss modules."""
from .module import Module
from ..functional import mse_loss
from ..tensor import Tensor


class MSELoss(Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("invalid reduction")
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return mse_loss(input, target, reduction=self.reduction)

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}'"
