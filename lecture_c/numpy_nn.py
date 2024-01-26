import numpy as np
import numpy.typing as npt


class NPLinear:
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.W = np.zeros((out_channels, in_channels))
        self.b = np.zeros(out_channels)
        self.W_grad = None
        self.b_grad = None

    def forward(self, x: npt.NDArray) -> npt.ArrayLike:
        return self.W @ np.transpose(x) + self.b

    def backward(self, x: npt.NDArray) -> npt.ArrayLike:
        pass

    def gd_update(self, lr: float) -> None:
        pass


class NPModel:
    def __init__(self) -> None:
        pass
