import matplotlib.pyplot as plt
import numpy as np

plt.style.use("_mpl-gallery")


def generate_data(num):
    x = np.linspace(0, 2 * np.pi, num=num)
    y = np.sin(x) + np.random.normal(loc=0, scale=0.1, size=num)
    return (x, y)


def fit_poly(x_train, y_train, k=1):
    assert k == 1
    # create design matrix X with added ones
    X = np.stack((x_train, np.ones_like(x_train)), axis=1)

    # compute weight matrix from formula
    W = y_train @ X @ np.linalg.inv((np.transpose(X) @ X))
    return W


def mse_poly(x, y, W):
    pass


def create_polynomial(W):
    return lambda x: sum(w * pow(x, n) for n, w in enumerate(W))


def main():
    # generate training and test data
    x_training_data, y_training_data = generate_data(15)
    x_test_data, y_test_data = generate_data(10)

    # plot
    fig, ax = plt.subplots()

    ax.plot(x_training_data, y_training_data)

    # ax.set(xlim=(0, 2 * np.pi), ylim=(1.5, 1.5))

    plt.show()


if __name__ == "__main__":
    main()
