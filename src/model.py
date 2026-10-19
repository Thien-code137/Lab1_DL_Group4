"""TV2 — kiến trúc MLP; import cần torch sau khi cài môi trường."""
from torch import nn

class MLP(nn.Module):
    def __init__(self, hidden_units=128):
        super().__init__()
        # TODO TV2: Flatten -> Linear(784, hidden_units) -> ReLU -> Linear(hidden_units, 10).
        # Không thêm Softmax; không hard-code batch size.
        raise NotImplementedError("TV2: triển khai MLP.__init__")

    def forward(self, images):
        """images float32 [B,1,28,28] -> logits [B,10]."""
        raise NotImplementedError("TV2: triển khai MLP.forward")
