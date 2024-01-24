import matplotlib.pyplot as plt
import numpy as np

plt.style.use("_mpl-gallery")


def generate_data(num, domain=(0, 2 * np.pi)):
    x = np.linspace(domain[0], domain[1], num=num)
    y = np.sin(x) + np.random.normal(loc=0, scale=0.1, size=num)
    return (x, y)


def fit_poly(x_train, y_train, k=1):
    X = np.transpose(np.array([x_train**n for n in range(k + 1)]))  # detail matrix
    return y_train @ X @ np.linalg.inv((np.transpose(X) @ X))


def mse_poly(x, y, W):
    fitted_poly = create_polynomial(W)
    fitted_y = fitted_poly(x)
    squared_error = (y - fitted_y) * (y - fitted_y)
    mse = np.average(squared_error)
    return mse


def create_polynomial(W):
    return lambda x: sum(w * pow(x, n) for n, w in enumerate(W))


def ridge_fit_poly(x_train, y_train, k, lamb):
    X = np.transpose(np.array([x_train**n for n in range(k + 1)]))  # detail matrix
    return (
        np.transpose(y_train)
        @ X
        @ np.linalg.inv(np.transpose(X) @ X + (lamb * np.eye(len(x_train))))
    )


def perform_cv(x, y, k, lamb, folds):
    pass


def main():
    # generate training and test data
    x_training_data, y_training_data = generate_data(5)
    x_test_data, y_test_data = generate_data(10)

    W = fit_poly(x_training_data, y_training_data, 5)

    x_fitted_data = np.linspace(0, 2 * np.pi, 200)
    fitted_polynomial = create_polynomial(W)
    y_fitted_data = fitted_polynomial(x_fitted_data)

    training_mse = mse_poly(x_training_data, y_training_data, W)

    # plot
    fig, ax = plt.subplots()

    (training_line,) = ax.plot(x_training_data, y_training_data)
    (fitted_line,) = ax.plot(x_fitted_data, y_fitted_data)

    ax.legend([training_line, fitted_line], [f"training data, MSE={training_mse}", "fitted data"])
    # ax.set(xlim=(0, 2 * np.pi), ylim=(1.5, 1.5))

    plt.show()

    x_training_data, y_training_data = generate_data(5)
    x_test_data, y_test_data = generate_data(10)
    k_values = np.arange(1, 16)
    log_mse_values = np.zeros_like(k_values)
    for k in k_values:
        W_k = fit_poly(x_training_data, y_training_data, k)
        log_mse_values[k - 1] = np.log10(mse_poly(x_training_data, y_training_data, W_k))

    fig, ax = plt.subplots()
    ax.plot(k_values, log_mse_values)

    plt.show()


if __name__ == "__main__":
    main()
