import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import math
from argparse import Namespace
import sys


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


# convert polynomial coefficients to a polynomial function
def generate_polynomial(W: torch.Tensor) -> "function":
    return lambda x: W @ featurize(x).T


# plot training data, ground truth, and model prediction
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
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    lr: float = 0.00001,
    step_count: int = 100,
    weight: torch.Tensor = None,
    momentum: float = 0,
    optimizer_type: str = None,
):
    # initialize NN
    if weight is None:
        weight = torch.ones(4)
    x = featurize(x_train)  # x features = [1, x, x^2, x^3]
    model = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        model.weight[...] = weight[...]

    # optimizer selection
    if optimizer_type is None or optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == "LBFGS":
        optimizer = optim.LBFGS(model.parameters(), lr=lr)
    else:
        print("unknown optimizer", file=sys.stderr)
        raise ValueError

    # optimizer/loss setup
    loss_func = nn.MSELoss()
    preds = model(x)
    targets = y_train.reshape((-1, 1))
    loss = loss_func(preds, targets)

    losses = torch.zeros(step_count)
    step_counts = torch.arange(1, step_count + 1)

    # optimizer loop
    for i in step_counts:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        preds = model(x)
        loss = loss_func(preds, targets)
        losses[i - 1] = loss.item()

    # results
    return model, losses


# search a range of learning rates (and momentums) to find best model after 100 steps
def lr_tuning(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    learning_rates: torch.Tensor,
    momentums: torch.Tensor = None,
    weight: torch.Tensor = None,
    optimizer_type: str = None,
) -> Namespace:
    # if no momentum is specified, use momentum=0
    if momentums is None:
        momentums = torch.zeros(1, dtype=float)

    # learning rate (and momentum) tuning
    results = dict()  # dict of form MSE : Namespace(model, lr, momentum)
    for lr, momentum in torch.cartesian_prod(learning_rates, momentums):
        # if momentum was set to None, the nested loops reduce to a loop only over lr's
        # if weight is set to None, the default [1, 1, 1, 1] is used
        model, losses = train_model(
            x_train,
            y_train,
            lr=lr.item(),
            weight=weight,
            momentum=momentum,
            optimizer_type=optimizer_type,
        )
        loss = losses[-1]  # the loss for the current model
        results[loss] = Namespace(model=model, lr=lr, momentum=momentum)

    # best model found during tuning
    best_mse = min(results.keys())
    best_model = results[best_mse].model
    best_lr = results[best_mse].lr.item()
    best_momentum = results[best_mse].momentum.item()
    best_results = Namespace(model=best_model, mse=best_mse, lr=best_lr, momentum=best_momentum)
    return best_results


def run_part_a():
    # learning rate tuning
    learning_rates = 10 ** torch.linspace(-15, 1, 160, dtype=float)
    tuning_results = lr_tuning(x_train, y_train, learning_rates)
    best_mse = tuning_results.mse
    best_model = tuning_results.model
    best_lr = tuning_results.lr

    # assert best_mse == 9.99654483795166
    # assert best_lr == 5.851882501969502e-05

    # plot_data(x_train, y_train, best_model, best_mse)

    # loss vs step_count for tuned learning rate
    STEP_COUNT = 2_000
    model, losses = train_model(x_train, y_train, best_lr, STEP_COUNT)
    plt.plot(torch.arange(100, STEP_COUNT + 1), losses[99:], label="loss vs step count")
    plt.axhline(y=losses[-1], color="r", linestyle="dashed", label="final loss")
    plt.legend()
    plt.show()

    # assert best_mse == 0.1106143668293953 ~ 0.05
    plot_data(x_train, y_train, model, losses[-1])


def run_part_b_momentum():
    # learning rate tuning for weight = [1, 0.1, 0.01, 0.001]
    learning_rates = 10 ** torch.linspace(-7, -2, 40, dtype=float)
    momentums = 10 ** torch.linspace(-5, 5, 20, dtype=float)
    weight = torch.tensor([1.0, 0.1, 0.01, 0.001])

    tuning_results = lr_tuning(
        x_train, y_train, learning_rates=learning_rates, momentums=momentums, weight=weight
    )
    best_mse = tuning_results.mse
    best_model = tuning_results.model
    best_lr = tuning_results.lr
    best_momentum = tuning_results.momentum

    # assert best_mse == 0.5352210998535156
    # assert best_lr == 6.812920690579622e-05
    # assert best_momentum == 0.01

    # plot_data(x_train, y_train, best_model, best_mse)

    # loss vs step_count for tuned learning rate
    STEP_COUNT = 2_000
    model, losses = train_model(
        x_train,
        y_train,
        lr=best_lr,
        momentum=best_momentum,
        step_count=STEP_COUNT,
        weight=weight,
    )
    plt.plot(torch.arange(100, STEP_COUNT + 1), losses[99:], label="loss vs step count")
    plt.axhline(y=losses[-1], color="r", linestyle="dashed", label="final loss")
    plt.legend()
    plt.show()

    best_model = model
    best_mse = losses[-1]  # ~ 0.03
    plot_data(x_train, y_train, best_model, best_mse)


def run_part_b_adam():
    pass


def run_part_b_bfgs():
    pass


if __name__ == "__main__":
    # run_part_a()
    run_part_b_momentum()
