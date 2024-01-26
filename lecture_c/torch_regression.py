import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt


N_TRAIN = 15
SIGMA_NOISE = 0.1
K = 3  # order of polynomial regression

torch.manual_seed(0xDEADBEEF)
x_train = torch.rand(N_TRAIN) * 2 * torch.pi
y_train = torch.sin(x_train) + torch.randn(N_TRAIN) * SIGMA_NOISE


# convert training data to detail matrix
def featurize(x_train: torch.Tensor) -> torch.Tensor:
    X = torch.ones((x_train.shape[0], 3 + 1))
    for i in range(1, 3 + 1):
        X[:, i] = x_train**i
    return X


# closed form regression
def cf_regression(x_train: torch.Tensor, y_train: torch.Tensor) -> torch.Tensor:
    X = featurize(x_train)
    return y_train @ X @ torch.linalg.inv(X.T @ X)


def generate_polynomial(W: torch.Tensor) -> "function":
    return lambda x: W @ featurize(x).T


def plot_data(
    x_train: torch.Tensor, y_train: torch.Tensor, y_pred: torch.Tensor, final_loss: float
) -> None:
    x_true = torch.linspace(0, 2 * torch.pi, 200)
    y_true = torch.sin(x_true)

    W_cf = cf_regression(x_train, y_train)
    cf_poly = generate_polynomial(W_cf)
    y_cf_regression = cf_poly(x_true)

    plt.plot(x_true, y_true, label="ground truth")
    plt.scatter(x_train, y_train, label="training data")
    plt.plot(x_true, y_cf_regression, label="closed form regression")
    plt.scatter(x_train, y_pred, label=f"SGD regression, MSE={final_loss}")

    plt.legend()
    plt.show()


def run_part_a():
    x = featurize(x_train)
    model = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        model.weight[...] = 1
    loss_func = nn.MSELoss()

    targets = y_train.reshape((-1, 1))

    sgd = optim.SGD(model.parameters(), lr=0.00001)

    preds = model(x)
    loss = loss_func(preds, targets)

    for i in range(100):
        loss.backward()
        sgd.step()
        sgd.zero_grad()
        preds = model(x)
        loss = loss_func(preds, targets)

    final_pred = preds.detach()
    final_loss = loss.item()
    plot_data(x_train, y_train, final_pred, final_loss)


if __name__ == "__main__":
    run_part_a()
