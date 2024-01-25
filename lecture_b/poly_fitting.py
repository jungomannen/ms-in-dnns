import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
    y_transpose = y_train.reshape((1, -1))  # convert to (1, N) array, i.e. transpose of y
    return y_transpose @ X @ np.linalg.inv((np.transpose(X) @ X) + (lamb * np.eye(k + 1)))


def perform_cv(x, y, k, lamb, folds):
    # split data
    x_folds = np.split(x, folds)
    y_folds = np.split(y, folds)

    fold_mse = np.zeros(folds)
    for n in range(folds):
        # partition data
        x_train = np.concatenate(x_folds[:n] + x_folds[n + 1 :])  # x with n:th fold removed
        y_train = np.concatenate(y_folds[:n] + y_folds[n + 1 :])
        x_test = x_folds[n]
        y_test = y_folds[n]

        # perform ridge regression
        W = ridge_fit_poly(x_train, y_train, k, lamb)
        fold_mse[n] = mse_poly(x_test, y_test, W)
    return np.average(fold_mse)


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


def cv_hyperparameter_search(num, k_values, lamb_values, print_logs=True):
    # generate data
    x_train, y_train = generate_data(120)

    # should probably create/find a function to compute this automatically...
    assert num == 120
    divisors = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60]

    # grid search tuning to find best k, lamb
    k_values = list(range(1, 21))
    lamb_values = 10 ** np.linspace(-5, 0, 20)
    mse_values = np.zeros((len(k_values), len(lamb_values)))
    for i, k in enumerate(k_values):
        if print_logs:
            print(f"iteration {i} out of {len(k_values)}")
        for j, lamb in enumerate(lamb_values):
            # compute average mse over all number of folds
            mse_estimate = 0.0
            for divisor in divisors:
                mse_k_lamb_div = perform_cv(x_train, y_train, k, lamb, divisor)
                mse_estimate += mse_k_lamb_div
            mse_estimate /= len(divisors)
            mse_values[i, j] = mse_estimate

    # best values found in grid search
    best_k_index, best_lamb_index = np.unravel_index(mse_values.argmin(), mse_values.shape)
    best_k = k_values[best_k_index]
    best_lamb = lamb_values[best_lamb_index]
    best_mse = mse_values[best_k_index, best_lamb_index]
    if print_logs:
        print(f"best k value = {best_k}, best lambda value = {best_lamb}, MSE={best_mse}")
    return best_k, best_lamb


def run_part_a_and_b():
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
    x_train, y_train = generate_data(15, domain=(0, 4 * np.pi))
    x_test, y_test = generate_data(1000, domain=(0, 4 * np.pi))
    domain = (0, 4 * np.pi)

    # test for k=5, lamb = 1
    # W = ridge_fit_poly(x_train, y_train, 5, 1)
    # mse = mse_poly(x_test, y_test, W)
    # plot_poly_fitting(x_train, y_train, W, mse, domain)

    # grid search tuning
    k_values = list(range(1, 21))
    lamb_values = 10 ** np.linspace(-5, 0, 20)
    log10_mse_values = np.zeros((len(k_values), len(lamb_values)))
    for i, k in enumerate(k_values):
        for j, lamb in enumerate(lamb_values):
            W_k_lamb = ridge_fit_poly(x_train, y_train, k, lamb)
            log10_mse = np.log10(mse_poly(x_test, y_test, W_k_lamb))
            log10_mse_values[i, j] = log10_mse

    # best values found in grid search
    best_k_index, best_lamb_index = np.unravel_index(
        log10_mse_values.argmin(), log10_mse_values.shape
    )
    best_k = k_values[best_k_index]
    best_lamb = lamb_values[best_lamb_index]
    best_mse = 10 ** log10_mse_values[best_k_index, best_lamb_index]
    best_W = ridge_fit_poly(x_train, y_train, best_k, best_lamb)
    print(f"best k value = {best_k}, best lambda value = {best_lamb}, MSE={best_mse}")

    # plot results
    ax = plt.subplot()
    im = ax.imshow(log10_mse_values)
    plt.xlabel("lambda")
    plt.ylabel("k")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()

    # plot optimal polynomial from grid search
    plot_poly_fitting(x_train, y_train, best_W, best_mse, domain)


def run_part_e():
    # do grid search to find best k, lamb
    k_values = list(range(1, 21))
    lamb_values = 10 ** np.linspace(-5, 0, 20)
    # best_k, best_lamb = cv_hyperparameter_search(120, k_values, lamb_values)

    # after running this once, I obtained:
    # best k value = 5, best lambda value = 0.04832930238571752, MSE=0.03568612470276438
    best_k = 5
    best_lamb = 0.04832930238571752

    # evaluate cross validation mse versus number of folds
    divisors = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60]  # divs of 120

    mse_vals = np.zeros((100, len(divisors)))
    for n in range(100):
        print(f"iteration {n} out of 100")
        x_data, y_data = generate_data(120)
        for i, divisor in enumerate(divisors):
            mse = perform_cv(x_data, y_data, best_k, best_lamb, divisor)
            mse_vals[n, i] = mse

    mse_means = np.average(mse_vals, axis=0)
    mse_standard_deviations = np.std(mse_vals, axis=0)

    # mse_means = np.array(
    #     [
    #         8.38757943e01,
    #         1.44424719e00,
    #         1.87242143e-01,
    #         5.42201325e-02,
    #         2.70221879e-02,
    #         1.84092112e-02,
    #         1.35707470e-02,
    #         1.23971302e-02,
    #         1.13834851e-02,
    #         1.09232429e-02,
    #         1.06711977e-02,
    #         1.05291514e-02,
    #         1.03977252e-02,
    #         1.02986414e-02,
    #     ]
    # )
    # mse_standard_deviations = np.array(
    #     [
    #         1.35617711e02,
    #         1.81139183e00,
    #         2.40226265e-01,
    #         5.17609013e-02,
    #         2.91175280e-02,
    #         8.63809458e-03,
    #         4.36642006e-03,
    #         3.15609359e-03,
    #         2.13211943e-03,
    #         1.90491065e-03,
    #         1.72336572e-03,
    #         1.55280189e-03,
    #         1.49772614e-03,
    #         1.42360097e-03,
    #     ]
    # )

    fig, ax = plt.subplots()
    log10_mse_means = np.log10(mse_means)
    upper_std = mse_means + mse_standard_deviations
    lower_std = mse_means - mse_standard_deviations

    # get rid of negative values
    lower_std[lower_std < 0] = 0

    log10_upper_std = np.log10(upper_std)
    log10_lower_std = np.log10(lower_std)

    (mse_line,) = ax.plot(divisors, log10_mse_means)
    (upper_std_line,) = ax.plot(divisors, log10_upper_std)
    (lower_std_line,) = ax.plot(divisors, log10_lower_std)

    ax.legend(
        [mse_line, upper_std_line, lower_std_line],
        ["log10(MSE)", "log10(MSE + sigma)", "log10(MSE - sigma)"],
    )
    plt.xlabel("number of folds")
    plt.ylabel("log10 mse")
    plt.show()


if __name__ == "__main__":
    run_part_a_and_b()
    run_part_c()
    run_part_d()
    run_part_e()
