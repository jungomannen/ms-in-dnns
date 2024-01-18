import matplotlib.pyplot as plt
import numpy as np

plt.style.use("_mpl-gallery")


def generate_data(num):
    x = np.linspace(0, 2 * np.pi, num=num)
    y = np.sin(x) + np.random.normal(loc=0, scale=0.1, size=num)
    return (x, y)


def fit_poly(x_train, y_train, k):
    pass


def mse_poly(x, y, W):
    pass


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
