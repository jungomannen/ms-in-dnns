import matplotlib.pyplot as plt
import numpy as np

plt.style.use("_mpl-gallery")


def generate_data(num, domain=None):
    if domain is None:
        domain = (0, 2 * np.pi)
    x = np.linspace(domain[0], domain[1], num=num) + np.random.normal(loc=0, scale=0.1, size=num)
    y = np.sin(x) + np.random.normal(loc=0, scale=0.1, size=num)
    return (x, y)


def fit_poly(x_train, y_train, k=1):
    X = np.transpose(np.array([x_train**n for n in range(k + 1)]))  # detail matrix
    y_train = y_train.reshape((1, -1))  # convert to (1, N) array, i.e. transpose of y
    return y_train @ X @ np.linalg.inv((np.transpose(X) @ X))


def mse_poly(x, y, W):
    fitted_poly = create_polynomial(W)
    fitted_y = fitted_poly(x)
    squared_error = (y - fitted_y) * (y - fitted_y)
    mse = np.average(squared_error)
    return mse


def create_polynomial(W):
    W = W.reshape(-1)  # convert to (N,) array
    return lambda x: sum(w * pow(x, n) for n, w in enumerate(W))


def poly(x, W):
    fitted_polynomial = create_polynomial(W)
    return fitted_polynomial(x)


def ridge_fit_poly(x_train, y_train, k, lamb):
    X = np.transpose(np.array([x_train**n for n in range(k + 1)]))  # detail matrix
    return (
        np.transpose(y_train)
        @ X
        @ np.linalg.inv(np.transpose(X) @ X + (lamb * np.eye(len(x_train))))
    )


def perform_cv(x, y, k, lamb, folds):
    pass


def plot_poly_fitting(x_train, y_train, W, mse, domain=None):
    if domain is None:
        domain = (0, 2 * np.pi)
    x_fitted_data = np.linspace(domain[0], domain[1], 200)
    fitted_polynomial = create_polynomial(W)
    y_fitted_data = fitted_polynomial(x_fitted_data)

    # plot
    fig, ax = plt.subplots()
    (training_line,) = ax.plot(x_train, y_train)
    (fitted_line,) = ax.plot(x_fitted_data, y_fitted_data)

    ax.legend([training_line, fitted_line], [f"training data, test MSE={mse}", "fitted data"])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def plot_k_search(k_values, error_data):
    plt.plot(k_values, error_data)
    plt.xlabel("k")
    plt.ylabel("log10 MSE")

    plt.show()


def run_part_a_and_b():
    """part 2.a and 2.b"""
    # generate training and test data
    x_training_data, y_training_data = generate_data(15)
    x_test_data, y_test_data = generate_data(10)

    # coefficients of fitted cubic polynomial
    W = fit_poly(x_training_data, y_training_data, 3)

    # mean squared error
    test_mse = mse_poly(x_test_data, y_test_data, W)

    # plot results
    plot_poly_fitting(x_training_data, y_training_data, W, test_mse)


def run_part_c():
    """part 2.c"""
    # create data
    x_training_data, y_training_data = generate_data(15, domain=(0, 4 * np.pi))
    x_test_data, y_test_data = generate_data(10, domain=(0, 4 * np.pi))

    # hyperparameter scan
    k_values = np.arange(1, 16)
    log_mse_values = np.zeros_like(k_values, float)
    for k in k_values:
        W_k = fit_poly(x_training_data, y_training_data, k)
        log_mse_values[k - 1] = np.log10(mse_poly(x_test_data, y_test_data, W_k))

    plot_k_search(k_values, log_mse_values)

    # conclusion: k = 7 seems to work well

    W = fit_poly(x_training_data, y_training_data, 7)
    mse = mse_poly(x_test_data, y_test_data, W)
    plot_poly_fitting(x_training_data, y_training_data, W, mse, domain=(0, 4 * np.pi))


def run_part_d():
    pass


def run_part_e():
    pass


def main():
    run_part_a_and_b()
    run_part_c()


if __name__ == "__main__":
    main()
