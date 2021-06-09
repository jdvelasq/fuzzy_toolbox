import matplotlib.pyplot as plt
import numpy as np
import progressbar
from sklearn.linear_model import LinearRegression


class Sugeno:
    def __init__(
        self, num_input_mfs=(3,), mftype="trimf", and_operator="min", seed=None
    ):
        self.num_input_mfs = num_input_mfs
        self.mftype = mftype
        self.and_operator = and_operator

        self.fuzzy_set_centers = None
        self.fuzzy_set_sigmas = None
        self.fuzzy_set_exponents = None
        self.antecedents = None
        self.rule_output = None

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    def __call__(self, X_antecedents, X_consequents):

        self.compute_memberships_by_rule_per_var(X_antecedents)
        self.compute_normalized_rule_weights(X_antecedents)
        self.compute_consequents(X_consequents)
        return self.compute_weighted_output()

    def compute_normalized_rule_weights(self, X):

        NPOINTS = len(X)
        NRULES = len(self.rules)
        NDIM = len(self.rules[0])

        if self.and_operator == "min":
            rule_memberships = self.memberships_by_rule_per_var.min(axis=1)

        if self.and_operator == "prod":
            rule_memberships = self.memberships_by_rule_per_var.prod(axis=1)

        rule_memberships = rule_memberships.reshape((NPOINTS, NRULES))
        sum_of_memberships = rule_memberships.sum(axis=1).reshape((NPOINTS, 1))
        sum_of_memberships = np.where(sum_of_memberships == 0, 1, sum_of_memberships)
        sum_of_memberships = np.tile(sum_of_memberships, (1, NRULES))
        rule_memberships = rule_memberships / sum_of_memberships
        self.rule_memberships = rule_memberships

    def compute_memberships_by_rule_per_var(self, X):

        NPOINTS = len(X)
        NRULES = len(self.rules)
        NDIM = len(self.rules[0])

        self.antecedent_fuzzy_sets = np.tile(self.rules, (NPOINTS, 1))

        self.data = np.repeat(X, NRULES, axis=0)
        self.memberships_by_rule_per_var = np.zeros(shape=(NPOINTS * NRULES, NDIM))

        if self.mftype == "trimf":
            self.compute_memberships_by_rule_per_var_trimf()

        if self.mftype == "gaussmf":
            self.compute_memberships_by_rule_per_var_gaussmf()

        if self.mftype == "gbellmf":
            self.compute_memberships_by_rule_per_var_gbellmf()

    def compute_memberships_by_rule_per_var_trimf(self):

        NDIM = len(self.rules[0])

        for i_dim in range(NDIM):

            n_sets = self.num_input_mfs[i_dim]

            fuzzy_sets = np.zeros(shape=(n_sets, 3))
            fuzzy_set_centers = self.fuzzy_set_centers[i_dim]

            for i_fuzzy_set in range(n_sets):
                fuzzy_sets[i_fuzzy_set, 0] = fuzzy_set_centers[i_fuzzy_set]
                fuzzy_sets[i_fuzzy_set, 1] = fuzzy_set_centers[i_fuzzy_set + 1]
                fuzzy_sets[i_fuzzy_set, 2] = fuzzy_set_centers[i_fuzzy_set + 2]

            fuzzy_sets_centers = fuzzy_sets[self.antecedent_fuzzy_sets[:, i_dim]]

            a = fuzzy_sets_centers[:, 0]
            b = fuzzy_sets_centers[:, 1]
            c = fuzzy_sets_centers[:, 2]

            x = self.data[:, i_dim]
            self.memberships_by_rule_per_var[:, i_dim] = np.maximum(
                0, np.minimum((x - a) / (b - a), (c - x) / (c - b))
            )

    def compute_memberships_by_rule_per_var_gaussmf(self):

        NDIM = len(self.rules[0])

        for i_dim in range(NDIM):

            n_sets = self.num_input_mfs[i_dim]

            fuzzy_set_centers = self.fuzzy_set_centers[i_dim]
            fuzzy_set_centers = fuzzy_set_centers[self.antecedent_fuzzy_sets[:, i_dim]]

            fuzzy_set_sigmas = self.fuzzy_set_sigmas[i_dim]
            fuzzy_set_sigmas = fuzzy_set_sigmas[self.antecedent_fuzzy_sets[:, i_dim]]

            x = self.data[:, i_dim]

            self.memberships_by_rule_per_var[:, i_dim] = np.exp(
                -(((x - fuzzy_set_centers) / fuzzy_set_sigmas) ** 2)
            )

    def compute_memberships_by_rule_per_var_gbellmf(self):

        NDIM = len(self.rules[0])

        for i_dim in range(NDIM):

            # n_sets = self.num_input_mfs[i_dim]

            fuzzy_set_centers = self.fuzzy_set_centers[i_dim]
            fuzzy_set_centers = fuzzy_set_centers[self.antecedent_fuzzy_sets[:, i_dim]]

            fuzzy_set_sigmas = self.fuzzy_set_sigmas[i_dim]
            fuzzy_set_sigmas = fuzzy_set_sigmas[self.antecedent_fuzzy_sets[:, i_dim]]

            fuzzy_set_exponents = self.fuzzy_set_exponents[i_dim]
            fuzzy_set_exponents = fuzzy_set_exponents[
                self.antecedent_fuzzy_sets[:, i_dim]
            ]

            x = self.data[:, i_dim]

            self.memberships_by_rule_per_var[:, i_dim] = 1 / (
                1
                + np.power(
                    ((x - fuzzy_set_centers) / fuzzy_set_sigmas) ** 2,
                    fuzzy_set_exponents,
                )
            )

    def compute_consequents(self, X):

        NRULES = len(self.rules)
        NPOINTS = len(X)

        output = np.matmul(X, np.transpose(self.coefs_))
        intercepts = self.intercept_.reshape((1, NRULES))
        intercepts = np.tile(intercepts, (NPOINTS, 1))
        self.consequents = output + intercepts

    def compute_weighted_output(self):

        output = self.rule_memberships * self.consequents
        return output.sum(axis=1).reshape(-1)

    def fit(self, X_antecedents, X_consequents, y, learning_rate=0.01, max_iter=10):

        self.create_structure(X_antecedents, X_consequents, y)

        history = {"loss": []}

        if max_iter > 0:

            factor = (learning_rate - 1e-5) / max_iter

            for _ in progressbar.progressbar(range(max_iter)):

                self.improve_fuzzysets(X_antecedents, X_consequents, y, learning_rate)
                self.improve_coefs(X_antecedents, X_consequents, y, learning_rate)
                self.improve_intercepts(X_antecedents, X_consequents, y, learning_rate)

                history["loss"].append(
                    np.mean((y - self.__call__(X_antecedents, X_consequents)) ** 2)
                )

                learning_rate = learning_rate - factor

        return history

    def improve_fuzzysets(self, X_antecedents, X_consequents, y, learning_rate):

        self.improve_fuzzysets_centers(X_antecedents, X_consequents, y, learning_rate)
        self.improve_fuzzysets_sigmas(X_antecedents, X_consequents, y, learning_rate)
        self.improve_fuzzysets_exponents(X_antecedents, X_consequents, y, learning_rate)

    def improve_fuzzysets_centers(self, X_antecedents, X_consequents, y, learning_rate):

        y_pred = self.__call__(X_antecedents, X_consequents)
        mse_base = np.mean((y - y_pred) ** 2)

        for i_var in range(len(self.num_input_mfs)):

            grad = np.zeros(shape=self.fuzzy_set_centers[i_var].shape)

            for i_comp in range(len(self.fuzzy_set_centers[i_var])):

                self.fuzzy_set_centers[i_var][i_comp] += 0.001

                y_pred = self.__call__(X_antecedents, X_consequents)
                mse_current = np.mean((y - y_pred) ** 2)
                grad[i_comp] = (mse_current - mse_base) / 0.001

                self.fuzzy_set_centers[i_var][i_comp] -= 0.001

            if np.linalg.norm(grad) > 0.0:
                grad = grad / np.linalg.norm(grad)

            self.fuzzy_set_centers[i_var] = (
                self.fuzzy_set_centers[i_var] - learning_rate * grad
            )

    def improve_fuzzysets_sigmas(self, X_antecedents, X_consequents, y, learning_rate):

        if self.fuzzy_set_sigmas is None:
            return

        y_pred = self.__call__(X_antecedents, X_consequents)
        mse_base = np.mean((y - y_pred) ** 2)

        for i_var in range(len(self.num_input_mfs)):

            grad = np.zeros(shape=self.fuzzy_set_sigmas[i_var].shape)

            for i_comp in range(len(self.fuzzy_set_centers[i_var])):

                self.fuzzy_set_sigmas[i_var][i_comp] += 0.001

                y_pred = self.__call__(X_antecedents, X_consequents)
                mse_current = np.mean((y - y_pred) ** 2)
                grad[i_comp] = (mse_current - mse_base) / 0.001

                self.fuzzy_set_sigmas[i_var][i_comp] -= 0.001

            if np.linalg.norm(grad) > 0.0:
                grad = grad / np.linalg.norm(grad)

            self.fuzzy_set_sigmas[i_var] = (
                self.fuzzy_set_sigmas[i_var] - learning_rate * grad
            )

    def improve_fuzzysets_exponents(
        self, X_antecedents, X_consequents, y, learning_rate
    ):

        if self.fuzzy_set_exponents is None:
            return

        y_pred = self.__call__(X_antecedents, X_consequents)
        mse_base = np.mean((y - y_pred) ** 2)

        for i_var in range(len(self.num_input_mfs)):

            grad = np.zeros(shape=self.fuzzy_set_exponents[i_var].shape)

            for i_comp in range(len(self.fuzzy_set_exponents[i_var])):

                self.fuzzy_set_exponents[i_var][i_comp] += 0.001

                y_pred = self.__call__(X_antecedents, X_consequents)
                mse_current = np.mean((y - y_pred) ** 2)
                grad[i_comp] = (mse_current - mse_base) / 0.001

                self.fuzzy_set_exponents[i_var][i_comp] -= 0.001

            if np.linalg.norm(grad) > 0.0:
                grad = grad / np.linalg.norm(grad)

            self.fuzzy_set_exponents[i_var] = (
                self.fuzzy_set_exponents[i_var] - learning_rate * grad
            )

    def improve_intercepts(self, X_antecedents, X_consequents, y, learning_rate):

        y_pred = self.__call__(X_antecedents, X_consequents)
        mse_base = np.mean((y - y_pred) ** 2)

        grad = np.zeros(shape=self.intercept_.shape)

        for i_row in range(self.intercept_.shape[0]):

            self.intercept_[i_row] += 0.001
            y_pred = self.__call__(X_antecedents, X_consequents)
            mse_current = np.mean((y - y_pred) ** 2)
            grad[i_row] = (mse_current - mse_base) / 0.001
            self.intercept_[i_row] -= 0.001

        if np.linalg.norm(grad) > 0.0:
            grad = grad / np.linalg.norm(grad)

        self.intercept_ = self.intercept_ - learning_rate * grad

    def improve_coefs(self, X_antecedents, X_consequents, y, learning_rate):

        y_pred = self.__call__(X_antecedents, X_consequents)
        mse_base = np.mean((y - y_pred) ** 2)

        grad = np.zeros(shape=self.coefs_.shape)

        for i_row in range(self.coefs_.shape[0]):
            for i_col in range(self.coefs_.shape[1]):

                self.coefs_[i_row, i_col] += 0.001

                y_pred = self.__call__(X_antecedents, X_consequents)
                mse_current = np.mean((y - y_pred) ** 2)

                grad[i_row, i_col] = (mse_current - mse_base) / 0.001

                self.coefs_[i_row, i_col] -= 0.001

        if np.linalg.norm(grad) > 0.0:
            grad = grad / np.linalg.norm(grad)

        self.coefs_ = self.coefs_ - learning_rate * grad

    def create_structure(self, X_antecedents, X_consequents, y):

        self.create_rules()
        self.create_antecedents(X_antecedents)
        self.create_consequents(X_consequents, y)

    def create_rules(self):
        def connect(sets):
            #
            # Esta es una función auxiliar recursiva
            #
            if len(sets) == 1:
                return [[i] for i in range(sets[0])]
            else:
                cur = sets[0]
                tail = sets[1:]
                return [[i] + e for e in connect(tail) for i in range(cur)]

        self.rules = connect(self.num_input_mfs)

    def create_antecedents(self, X_antecedents):

        x_min = X_antecedents.min(axis=0)
        x_max = X_antecedents.max(axis=0)

        self.x_min = x_min
        self.x_max = x_max

        self.fuzzy_set_centers = []
        if self.mftype == "gaussmf":
            self.fuzzy_set_sigmas = []
        if self.mftype == "gbellmf":
            self.fuzzy_set_sigmas = []
            self.fuzzy_set_exponents = []

        for i_var in range(len(x_min)):

            n_sets = self.num_input_mfs[i_var]
            delta_x = (x_max[i_var] - x_min[i_var]) / (n_sets - 1)

            if self.mftype == "trimf":

                self.fuzzy_set_centers.append(
                    np.linspace(
                        start=x_min[i_var] - delta_x,
                        stop=x_max[i_var] + delta_x,
                        num=n_sets + 2,
                    )
                )
            else:
                self.fuzzy_set_centers.append(
                    np.linspace(
                        start=x_min[i_var],
                        stop=x_max[i_var],
                        num=n_sets,
                    )
                )

            if self.mftype == "gaussmf":
                self.fuzzy_set_sigmas.append(np.array([delta_x / 2.0] * n_sets))

            if self.mftype == "gbellmf":
                self.fuzzy_set_sigmas.append(np.array([delta_x / 2.0] * n_sets))
                self.fuzzy_set_exponents.append(np.array([1.0] * n_sets))

    def create_consequents(self, X_consequents, y):

        NRULES = np.prod(self.num_input_mfs)

        m = LinearRegression()
        m.fit(X_consequents, y)
        self.coefs_ = m.coef_.reshape((1, len(m.coef_)))
        self.coefs_ = np.tile(self.coefs_, (NRULES, 1))
        self.intercept_ = np.array([m.intercept_] * NRULES)

    def plot_fuzzysets(self, i_var, figsize=(8, 3)):
        #
        def plot_trimf():

            fuzzy_set_centers = self.fuzzy_set_centers[i_var]

            fuzzy_sets = np.zeros(shape=(n_sets, 3))

            for i_fuzzy_set in range(n_sets):
                fuzzy_sets[i_fuzzy_set, 0] = fuzzy_set_centers[i_fuzzy_set]
                fuzzy_sets[i_fuzzy_set, 1] = fuzzy_set_centers[i_fuzzy_set + 1]
                fuzzy_sets[i_fuzzy_set, 2] = fuzzy_set_centers[i_fuzzy_set + 2]

            for i_fuzzy_set in range(n_sets):

                a = fuzzy_sets[i_fuzzy_set, 0]
                b = fuzzy_sets[i_fuzzy_set, 1]
                c = fuzzy_sets[i_fuzzy_set, 2]

                membership = np.maximum(
                    0, np.minimum((x - a) / (b - a), (c - x) / (c - b))
                )
                plt.plot(x, membership)

        def plot_gaussmf():

            fuzzy_set_centers = self.fuzzy_set_centers[i_var]
            fuzzy_set_sigmas = self.fuzzy_set_sigmas[i_var]

            for i_fuzzy_set in range(n_sets):
                c = fuzzy_set_centers[i_fuzzy_set]
                s = fuzzy_set_sigmas[i_fuzzy_set]
                membership = np.exp(-(((x - c) / s) ** 2))
                plt.plot(x, membership)

        def plot_gbellmf():

            fuzzy_set_centers = self.fuzzy_set_centers[i_var]
            fuzzy_set_sigmas = self.fuzzy_set_sigmas[i_var]
            fuzzy_set_exponents = self.fuzzy_set_exponents[i_var]

            for i_fuzzy_set in range(n_sets):
                c = fuzzy_set_centers[i_fuzzy_set]
                s = fuzzy_set_sigmas[i_fuzzy_set]
                e = fuzzy_set_exponents[i_fuzzy_set]
                membership = 1 / (1 + np.power(((x - c) / s) ** 2, e))
                plt.plot(x, membership)

        n_sets = self.num_input_mfs[i_var]
        x_min = self.x_min[i_var]
        x_max = self.x_max[i_var]
        x = np.linspace(start=x_min, stop=x_max, num=100)

        plot_fn = {
            "trimf": plot_trimf,
            "gaussmf": plot_gaussmf,
            "gbellmf": plot_gbellmf,
        }[self.mftype]

        plt.figure(figsize=figsize)
        plot_fn()
        plt.ylim(-0.05, 1.05)
        plt.gca().spines["left"].set_color("lightgray")
        plt.gca().spines["bottom"].set_color("gray")
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)


# x1 = np.linspace(start=0, stop=10, num=100)
# x2 = np.random.uniform(0, 10, 100)
# y1 = np.sin(x1) + np.cos(x1)
# y2 = (y1) / np.exp(x1)

# import matplotlib.pyplot as plt
# import pandas as pd


# X = pd.DataFrame({"x1": x1, "x2": x2})

# m = Sugeno(num_input_mfs=(3, 3), mftype="gbellmf")

# m.fit(X.values, X.values, y2, learning_rate=0.1, max_iter=50)

# m.plot_fuzzysets(0)
# m.plot_fuzzysets(1)
# np.mean((y2 - m(X.values, X.values)) ** 2)

# m(X.values)

# y_pred = m(X.values)
# plt.plot(y2, "-k")
# plt.plot(y_pred, "-r")
