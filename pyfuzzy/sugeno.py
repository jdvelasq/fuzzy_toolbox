import matplotlib.pyplot as plt
import numpy as np
import progressbar


class Sugeno:
    def __init__(
        self, num_input_mfs=(3,), mftype="trimf", and_operator="min", seed=None
    ):
        self.num_input_mfs = num_input_mfs
        self.mftype = mftype
        self.and_operator = and_operator

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

        self.fuzzy_set_centers = None
        self.fuzzy_set_sigmas = None
        self.fuzzy_set_exponents = None

        self.antecedents = None
        self.rule_output = None

    def compute_memberships_by_rule(self, X):

        NPOINTS = len(X)
        NRULES = len(self.rules)
        NDIM = len(self.rules[0])

        fuzzy_index = np.tile(self.rules, (NPOINTS, 1))

        data = np.repeat(X, NRULES, axis=0)
        rule_memberships = np.zeros(shape=(NPOINTS * NRULES, NDIM))

        for i_dim in range(NDIM):

            n_sets = self.num_input_mfs[i_dim]

            if self.mftype == "trimf":

                fuzzy_sets = np.zeros(shape=(n_sets, 3))
                fuzzy_set_centers = self.fuzzy_set_centers[i_dim]

                for i_fuzzy_set in range(n_sets):
                    fuzzy_sets[i_fuzzy_set, 0] = fuzzy_set_centers[i_fuzzy_set]
                    fuzzy_sets[i_fuzzy_set, 1] = fuzzy_set_centers[i_fuzzy_set + 1]
                    fuzzy_sets[i_fuzzy_set, 2] = fuzzy_set_centers[i_fuzzy_set + 2]

                fuzzy_sets_centers = fuzzy_sets[fuzzy_index[:, i_dim]]

                a = fuzzy_sets_centers[:, 0]
                b = fuzzy_sets_centers[:, 1]
                c = fuzzy_sets_centers[:, 2]

                x = data[:, i_dim]
                membership = np.maximum(
                    0, np.minimum((x - a) / (b - a), (c - x) / (c - b))
                )

            if self.mftype == "gaussmf":

                fuzzy_set_centers = self.fuzzy_set_centers[i_dim]
                fuzzy_set_centers = fuzzy_set_centers[fuzzy_index[:, i_dim]]

                fuzzy_set_sigmas = self.fuzzy_set_sigmas[i_dim]
                fuzzy_set_sigmas = fuzzy_set_sigmas[fuzzy_index[:, i_dim]]

                x = data[:, i_dim]

                membership = np.exp(
                    -(((x - fuzzy_set_centers) / fuzzy_set_sigmas) ** 2)
                )

            if self.mftype == "gbellmf":

                fuzzy_set_centers = self.fuzzy_set_centers[i_dim]
                fuzzy_set_centers = fuzzy_set_centers[fuzzy_index[:, i_dim]]

                fuzzy_set_sigmas = self.fuzzy_set_sigmas[i_dim]
                fuzzy_set_sigmas = fuzzy_set_sigmas[fuzzy_index[:, i_dim]]

                fuzzy_set_exponents = self.fuzzy_set_exponents[i_dim]
                fuzzy_set_exponents = fuzzy_set_exponents[fuzzy_index[:, i_dim]]

                x = data[:, i_dim]

                membership = 1 / (
                    1
                    + np.power(
                        (x - fuzzy_set_centers) / fuzzy_set_sigmas,
                        np.abs(2 * fuzzy_set_exponents),
                    )
                )

            rule_memberships[:, i_dim] = membership

        if self.and_operator == "min":
            rule_memberships = rule_memberships.min(axis=1)
        if self.and_operator == "prod":
            rule_memberships = rule_memberships.prod(axis=1)
        rule_memberships = rule_memberships.reshape((NPOINTS, NRULES))
        sum_of_memberships = rule_memberships.sum(axis=1).reshape((NPOINTS, 1))
        sum_of_memberships = np.where(sum_of_memberships == 0, 1, sum_of_memberships)
        sum_of_memberships = np.tile(sum_of_memberships, (1, NRULES))
        rule_memberships = rule_memberships / sum_of_memberships
        self.rule_memberships = rule_memberships

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

    def __call__(self, X_antecedents, X_consequents):

        self.compute_memberships_by_rule(X_antecedents)
        self.compute_consequents(X_consequents)
        return self.compute_weighted_output()

    def compute_least_squares(self, X, y):

        NRULES = len(self.rules)
        NPOINTS = len(X)
        NVARS = len(self.rules[0])

        self.compute_memberships_by_rule(X)
        memberships = self.rule_memberships.copy()
        memberships = np.repeat(memberships, NVARS, axis=1)
        x_ = np.tile(X, (1, NRULES))
        A = np.append(memberships * x_, memberships, axis=1)

        invAtA = np.linalg.pinv(np.matmul(np.transpose(A), A))
        AtB = np.matmul(np.transpose(A), y.reshape((NPOINTS, 1)))
        solution = np.matmul(invAtA, AtB)
        solution = solution.reshape(-1)

        self.intercepts_ = solution[-NRULES:]

        solution = solution[:-NRULES]
        coefs = solution[:-NRULES].reshape((NRULES, NVARS))
        self.coefs_ = coefs

    def call(self, X, y):
        self.compute_memberships_by_rule(X)
        self.compute_least_squares(X, y)
        self.compute_consequents(X)
        return self.compute_weighted_output()

    def fit(self, X_antecedents, X_consequents, y, learning_rate=0.01, max_iter=10):

        self.create_structure(X_antecedents, X_consequents, y)

        history = {"loss": []}

        if max_iter > 0:

            for _ in progressbar.progressbar(range(max_iter)):

                self.improve_fuzzysets(X_antecedents, X_consequents, y, learning_rate)
                self.improve_coefs(X_antecedents, X_consequents, y, learning_rate)
                self.improve_intercepts(X_antecedents, X_consequents, y, learning_rate)

                history["loss"].append(
                    np.mean((y - self.__call__(X_antecedents, X_consequents)) ** 2)
                )

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

        self.coefs_ = self.coefs_ - learning_rate * grad

    def create_structure(self, X_antecedents, X_consequents, y):

        self.create_rules()
        self.create_antecedents(X_antecedents)
        self.create_consequents(X_consequents)

        from sklearn.linear_model import LinearRegression

        NRULES = len(self.rules)

        m = LinearRegression()
        m.fit(X_consequents, y)
        self.coefs_ = m.coef_.reshape((1, len(m.coef_)))
        self.coefs_ = np.tile(self.coefs_, (NRULES, 1))
        self.intercept_ = np.array([m.intercept_] * NRULES)

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

    def create_consequents(self, X_consequents):
        n_vars = X_consequents.shape[1]
        n_rules = np.prod(self.num_input_mfs)
        self.coefs_ = self.rng.normal(loc=0, scale=0.1, size=(n_rules, n_vars))
        self.intercept_ = self.rng.normal(loc=0, scale=0.1, size=n_rules)

    def plot_fuzzysets(self, i_var, figsize=(8, 3)):

        if self.mftype == "trimf":
            return self.plot_fuzzysets_trimf(i_var=i_var, figsize=figsize)
        if self.mftype == "gaussmf":
            return self.plot_fuzzysets_gaussmf(i_var=i_var, figsize=figsize)
        if self.mftype == "gbellmf":
            return self.plot_fuzzysets_gbellmf(i_var=i_var, figsize=figsize)

        plt.ylim(-0.05, 1.05)
        plt.gca().spines["left"].set_color("lightgray")
        plt.gca().spines["bottom"].set_color("gray")
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

    def plot_fuzzysets_gaussmf(self, i_var, figsize=(8, 3)):

        n_sets = self.num_input_mfs[i_var]
        x_min = self.x_min[i_var]
        x_max = self.x_max[i_var]
        x = np.linspace(start=x_min, stop=x_max, num=100)

        n_sets = self.num_input_mfs[i_var]
        fuzzy_set_centers = self.fuzzy_set_centers[i_var]
        fuzzy_set_sigmas = self.fuzzy_set_sigmas[i_var]

        plt.figure(figsize=figsize)
        for i_fuzzy_set in range(n_sets):
            c = fuzzy_set_centers[i_fuzzy_set]
            s = fuzzy_set_sigmas[i_fuzzy_set]
            membership = np.exp(-(((x - c) / s) ** 2))
            plt.plot(x, membership)

    def plot_fuzzysets_gbellmf(self, i_var, figsize=(8, 3)):

        n_sets = self.num_input_mfs[i_var]
        x_min = self.x_min[i_var]
        x_max = self.x_max[i_var]
        x = np.linspace(start=x_min, stop=x_max, num=100)

        n_sets = self.num_input_mfs[i_var]
        fuzzy_set_centers = self.fuzzy_set_centers[i_var]
        fuzzy_set_sigmas = self.fuzzy_set_sigmas[i_var]
        fuzzy_set_exponents = self.fuzzy_set_exponents[i_var]

        plt.figure(figsize=figsize)
        for i_fuzzy_set in range(n_sets):
            c = fuzzy_set_centers[i_fuzzy_set]
            s = fuzzy_set_sigmas[i_fuzzy_set]
            e = fuzzy_set_exponents[i_fuzzy_set]
            membership = 1 / (1 + np.power((x - c) / s), np.abs(2 * e))
            plt.plot(x, membership)

    def plot_fuzzysets_trimf(self, i_var, figsize=(8, 3)):

        plt.figure(figsize=figsize)

        n_sets = self.num_input_mfs[i_var]
        x_min = self.x_min[i_var]
        x_max = self.x_max[i_var]
        x = np.linspace(start=x_min, stop=x_max, num=100)

        n_sets = self.num_input_mfs[i_var]
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

            membership = np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))
            plt.plot(x, membership)


# x1 = np.linspace(start=0, stop=10, num=100)
# x2 = np.random.uniform(0, 10, 100)
# y1 = np.sin(x1) + np.cos(x1)
# y2 = (y1) / np.exp(x1)

# import matplotlib.pyplot as plt
# import pandas as pd


# X = pd.DataFrame({"x1": x1, "x2": x2})

# m = Sugeno(num_input_mfs=(3, 3))

# m.fit(X.values, X.values, y2, learning_rate=0.01, max_iter=10)
# np.mean((y2 - m(X.values, X.values)) ** 2)

# m(X.values)

# y_pred = m(X.values)
# plt.plot(y2, "-k")
# plt.plot(y_pred, "-r")
