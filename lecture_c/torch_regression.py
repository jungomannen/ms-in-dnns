import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import math
from argparse import Namespace


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
    x_train: torch.Tensor, y_train: torch.Tensor, model: nn.Linear, final_loss: float
) -> None:
    x_true = torch.linspace(0, 2 * torch.pi, 200)
    y_true = torch.sin(x_true)

    W_cf = cf_regression(x_train, y_train)
    cf_poly = generate_polynomial(W_cf)
    y_cf_regression = cf_poly(x_true)

    W_nn = model.weight.detach().reshape(-1)
    nn_poly = generate_polynomial(W_nn)
    y_nn_regression = nn_poly(x_true)

    plt.plot(x_true, y_true, label="ground truth")
    plt.scatter(x_train, y_train, label="training data")
    plt.plot(x_true, y_cf_regression, label="closed form regression")
    plt.plot(x_true, y_nn_regression, label=f"SGD regression, MSE={final_loss}")

    plt.legend()
    plt.show()


def train_model(
    x_train: torch.Tensor, y_train: torch.Tensor, lr: float, step_count: int = 100
) -> (nn.Linear, float):
    x = featurize(x_train)
    model = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        model.weight[...] = 1
    loss_func = nn.MSELoss()

    targets = y_train.reshape((-1, 1))

    sgd = optim.SGD(model.parameters(), lr=lr)

    preds = model(x)
    loss = loss_func(preds, targets)

    for i in range(step_count):
        loss.backward()
        sgd.step()
        sgd.zero_grad()
        preds = model(x)
        loss = loss_func(preds, targets)

    # final_pred = preds.detach()
    final_loss = loss.item()
    return model, final_loss


def run_part_a():
    # learning rate tuning
    results = dict()  # dict of form MSE : Namespace(model, lr, step_count)
    learning_rates = 10 ** torch.arange(-15, 1, 0.1, dtype=float)
    for lr in learning_rates:
        model, loss = train_model(x_train, y_train, lr.item())
        if math.isnan(loss) or math.isinf(loss):
            break
        results[loss] = Namespace(model=model, lr=lr, step_count=100)

    # best model found during tuning
    best_mse = min(results.keys())
    best_model = results[best_mse].model
    best_lr = results[best_mse].lr.item()
    # best_preds = best_model(featurize(x_train)).detach()
    # print(best_lr)

    # plot_data(x_train, y_train, best_preds, best_mse)

    # loss vs step_count for tuned learning rate
    step_counts = 10 ** torch.arange(1, 5, 1, dtype=float)
    losses = torch.zeros_like(step_counts, dtype=float)
    for i, step_count in enumerate(step_counts):
        model, loss = train_model(x_train, y_train, best_lr, int(step_count))
        losses[i] = loss
    plt.plot(torch.log10(step_counts), torch.log10(losses), label="log10loss vs log10 step count")
    plt.legend()
    plt.show()

    best_model = model
    best_mse = loss
    plot_data(x_train, y_train, best_model, best_mse)


if __name__ == "__main__":
    run_part_a()
