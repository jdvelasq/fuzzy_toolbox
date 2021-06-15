"""
Sugeno fuzzy model
==============================================================================

"""

import matplotlib.pyplot as plt
import numpy as np
import progressbar
from sklearn.linear_model import LinearRegression


class Sugeno:
    """Creates a Sugeno inference system.

    Args:
        num_input_mfs (tuple of integers): Number of fuzzy sets for each variable in the antecedent.
        mftype (string): {"trimf"|"gaussmf"|"gbellmf"}. Type of memberships used.
        and_operator (string): {"min"|"prod"}. Operator used to compute the firing strength of the rules.
        seed (int, None): seed of the random number generator.

    """

    def __init__(
        self,
        num_input_mfs=(3,),
        mftype="trimf",
        and_operator="min",
        seed=None,
    ):
        self.num_input_mfs = num_input_mfs
        self.mftype = mftype
        self.and_operator = and_operator

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

        #
        # parameters
        #
        self.params = None

        #
        # internal structure
        #
        self.premises = None

    # =========================================================================
    #
    # Internal model structure
    #
    # =========================================================================
    def create_internal_structure(self, X_premises, X_consequences):
        """Creates the internal structure of the model."""

        def create_premises():
            #
            # Para un modelo con dos variables, 2 fuzzy sets para la
            # primera y tres para la segunda
            #
            # premises = [
            #    [0, 0],
            #    [0, 1],
            #    [0, 2],
            #    [1, 0],
            #    [2, 1],
            #    [3, 2],
            # ]
            #
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

            self.premises = connect(self.num_input_mfs)

        def create_mf_params():
            def create_trimf_params():

                self.params["a"] = []
                self.params["b"] = []
                self.params["c"] = []

                for i_var in range(X_premises.shape[1]):

                    n_sets = self.num_input_mfs[i_var]
                    delta_x = (self.x_max[i_var] - self.x_min[i_var]) / (n_sets - 1)

                    self.params["a"].append(
                        np.linspace(
                            start=self.x_min[i_var] - delta_x,
                            stop=self.x_max[i_var] - delta_x,
                            num=n_sets,
                        )
                    )

                    self.params["b"].append(
                        np.linspace(
                            start=self.x_min[i_var],
                            stop=self.x_max[i_var],
                            num=n_sets,
                        )
                    )

                    self.params["c"].append(
                        np.linspace(
                            start=self.x_min[i_var] + delta_x,
                            stop=self.x_max[i_var] + delta_x,
                            num=n_sets,
                        )
                    )

            def create_gaussmf_params():

                self.params["a"] = []
                self.params["b"] = []

                for i_var in range(X_premises.shape[1]):

                    n_sets = self.num_input_mfs[i_var]
                    delta_x = (self.x_max[i_var] - self.x_min[i_var]) / (n_sets - 1)

                    self.params["a"].append(
                        np.linspace(
                            start=self.x_min[i_var],
                            stop=self.x_max[i_var],
                            num=n_sets,
                        )
                    )

                    self.params["b"].append(np.array([delta_x / 2.0] * n_sets))

            def create_gbellmf_params():

                self.params["a"] = []
                self.params["b"] = []
                self.params["c"] = []

                for i_var in range(X_premises.shape[1]):

                    n_sets = self.num_input_mfs[i_var]
                    delta_x = (self.x_max[i_var] - self.x_min[i_var]) / (n_sets - 1)

                    self.params["a"].append(
                        np.linspace(
                            start=self.x_min[i_var],
                            stop=self.x_max[i_var],
                            num=n_sets,
                        )
                    )
                    self.params["b"].append(np.array([delta_x / 2.0] * n_sets))
                    self.params["c"].append(np.array([3.0] * n_sets))

            if self.mftype == "trimf":
                create_trimf_params()
            if self.mftype == "gaussmf":
                create_gaussmf_params()
            if self.mftype == "gbellmf":
                create_gbellmf_params()

        def create_consequents():

            NRULES = np.prod(self.num_input_mfs)
            if X_consequences is not None:
                NVARS = X_consequences.shape[1]
                self.params["coefs"] = self.rng.uniform(
                    low=-0.1, high=0.1, size=(NRULES, NVARS)
                )

            self.params["intercepts"] = self.rng.uniform(
                low=-0.1, high=0.1, size=NRULES
            )

        self.params = {}

        x_min = X_premises.min(axis=0)
        x_max = X_premises.max(axis=0)

        self.x_min = x_min
        self.x_max = x_max

        create_premises()
        create_mf_params()
        create_consequents()

    # =========================================================================
    #
    # Model prediction
    #
    # =========================================================================

    def compute_memberships_by_rule_per_var(self, X_premises):
        #
        def compute_trimf():

            n_var = X_premises.shape[1]

            for i_var in range(n_var):

                a = self.params["a"][i_var]
                b = self.params["b"][i_var]
                c = self.params["c"][i_var]

                a = a[self.premises_by_rule_per_var[:, i_var]]
                b = b[self.premises_by_rule_per_var[:, i_var]]
                c = c[self.premises_by_rule_per_var[:, i_var]]

                x = self.data[:, i_var]

                self.memberships_by_rule_per_var[:, i_var] = np.maximum(
                    0, np.minimum((x - a) / (b - a), (c - x) / (c - b))
                )

        def compute_gaussmf():

            n_var = X_premises.shape[1]

            for i_var in range(n_var):

                a = self.params["a"][i_var]
                b = self.params["b"][i_var]

                a = a[self.premises_by_rule_per_var[:, i_var]]
                b = b[self.premises_by_rule_per_var[:, i_var]]

                x = self.data[:, i_var]

                self.memberships_by_rule_per_var[:, i_var] = np.exp(
                    -(((x - a) / b) ** 2)
                )

        def compute_gbellmf():

            n_var = X_premises.shape[1]

            for i_var in range(n_var):

                a = self.params["a"][i_var]
                b = self.params["b"][i_var]
                c = self.params["c"][i_var]

                a = a[self.premises_by_rule_per_var[:, i_var]]
                b = b[self.premises_by_rule_per_var[:, i_var]]
                c = c[self.premises_by_rule_per_var[:, i_var]]

                x = self.data[:, i_var]

                self.memberships_by_rule_per_var[:, i_var] = 1 / (
                    1 + np.power(((x - a) / b) ** 2, c)
                )

        #
        # main body
        #
        NPOINTS = X_premises.shape[0]
        NRULES = len(self.premises)
        NVARS = len(self.premises[0])

        self.premises_by_rule_per_var = np.tile(self.premises, (NPOINTS, 1))

        self.data = np.repeat(X_premises, NRULES, axis=0)
        self.memberships_by_rule_per_var = np.zeros(shape=(NPOINTS * NRULES, NVARS))

        if self.mftype == "trimf":
            compute_trimf()

        if self.mftype == "gaussmf":
            compute_gaussmf()

        if self.mftype == "gbellmf":
            compute_gbellmf()

    def compute_normalized_firing_strenghts(self, n_points):

        NRULES = len(self.premises)

        if self.and_operator == "min":
            firing_strenghts = self.memberships_by_rule_per_var.min(axis=1)

        if self.and_operator == "prod":
            firing_strenghts = self.memberships_by_rule_per_var.prod(axis=1)

        firing_strenghts = firing_strenghts.reshape((n_points, NRULES))

        sum_of_firing_strenghts = firing_strenghts.sum(axis=1).reshape((n_points, 1))
        sum_of_firing_strenghts = np.where(
            sum_of_firing_strenghts == 0, 1, sum_of_firing_strenghts
        )
        sum_of_firing_strenghts = np.tile(sum_of_firing_strenghts, (1, NRULES))
        self.normalized_firing_strenghts = firing_strenghts / sum_of_firing_strenghts

    def compute_consequences(self, X_consequences):

        NRULES = len(self.premises)
        NPOINTS = X_consequences.shape[0]

        if "coefs" in self.params.keys():
            output = np.matmul(X_consequences, np.transpose(self.params["coefs"]))
        else:
            output = np.zeros(shape=(NPOINTS, 1))

        intercepts = self.params["intercepts"].reshape((1, NRULES))
        intercepts = np.tile(intercepts, (NPOINTS, 1))
        self.consequences = output + intercepts

    def compute_weighted_output(self):

        output = self.normalized_firing_strenghts * self.consequences
        return output.sum(axis=1).reshape(-1)

    def __call__(self, X_premises, X_consequences):

        if self.params is None:
            self.create_internal_structure(X_premises, X_consequences)

        self.compute_memberships_by_rule_per_var(X_premises)
        self.compute_normalized_firing_strenghts(n_points=X_premises.shape[0])
        self.compute_consequences(X_consequences)
        return self.compute_weighted_output()

    # =========================================================================
    #
    # Model fiting
    #
    # =========================================================================

    def fit(
        self,
        X_premises,
        X_consequences,
        y,
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=10,
        warm_start=False,
        batch_size="auto",
        shuffle=True,
    ):

        if self.params is None or warm_start is False:
            self.create_internal_structure(X_premises, X_consequences)

            NRULES = len(self.premises)
            m = LinearRegression()
            m.fit(X_consequences, y)
            coefs = m.coef_.reshape((1, len(m.coef_)))
            self.params["coefs"] = np.tile(coefs, (NRULES, 1))
            self.params["intercepts"] = np.array([m.intercept_] * NRULES)

        X_premises = X_premises.copy()
        X_consequences = X_consequences.copy()
        n_samples = X_premises.shape[0]

        if isinstance(batch_size, str) and batch_size == "auto":
            batch_size = min(200, X_premises.shape[0])

        if batch_size is None:
            batch_size = n_samples

        indexes = np.arange(n_samples)

        history = {"loss": []}

        if max_iter > 0:

            lr = learning_rate_init

            for iter in progressbar.progressbar(range(max_iter)):

                if shuffle is True and n_samples > 1:
                    self.rng.shuffle(indexes)

                mse_base_lr = np.mean(
                    (y - self.__call__(X_premises, X_consequences)) ** 2
                )

                for i_batch in range(0, n_samples, batch_size):

                    batch_indexes = indexes[i_batch : i_batch + batch_size]

                    for param in self.params.keys():

                        if isinstance(self.params[param], list):

                            for i_array, _ in enumerate(self.params[param]):
                                for i, _ in enumerate(self.params[param][i_array]):

                                    mse_base = np.mean(
                                        (
                                            y[batch_indexes]
                                            - self.__call__(
                                                X_premises[batch_indexes, :],
                                                X_consequences[batch_indexes, :],
                                            )
                                        )
                                        ** 2
                                    )
                                    self.params[param][i_array][i] += 0.001
                                    mse_current = np.mean(
                                        (
                                            y[batch_indexes]
                                            - self.__call__(
                                                X_premises[batch_indexes, :],
                                                X_consequences[batch_indexes, :],
                                            )
                                        )
                                        ** 2
                                    )
                                    grad = (mse_current - mse_base) / 0.001
                                    self.params[param][i_array][i] -= 0.001
                                    self.params[param][i_array][i] -= lr * grad

                        if (
                            isinstance(self.params[param], np.ndarray)
                            and len(self.params[param].shape) == 2
                        ):

                            for i_row in range(self.params[param].shape[0]):

                                for i_col in range(self.params[param].shape[1]):

                                    mse_base = np.mean(
                                        (
                                            y[batch_indexes]
                                            - self.__call__(
                                                X_premises[batch_indexes, :],
                                                X_consequences[batch_indexes, :],
                                            )
                                        )
                                        ** 2
                                    )
                                    self.params[param][i_row, i_col] += 0.001
                                    mse_current = np.mean(
                                        (
                                            y[batch_indexes]
                                            - self.__call__(
                                                X_premises[batch_indexes, :],
                                                X_consequences[batch_indexes, :],
                                            )
                                        )
                                        ** 2
                                    )
                                    grad = (mse_current - mse_base) / 0.001
                                    self.params[param][i_row, i_col] -= 0.001
                                    self.params[param][i_row, i_col] -= lr * grad

                        if (
                            isinstance(self.params[param], np.ndarray)
                            and len(self.params[param].shape) == 1
                        ):

                            for i in range(self.params[param].shape[0]):

                                mse_base = np.mean(
                                    (
                                        y[batch_indexes]
                                        - self.__call__(
                                            X_premises[batch_indexes, :],
                                            X_consequences[batch_indexes, :],
                                        )
                                    )
                                    ** 2
                                )
                                self.params[param][i] += 0.001
                                mse_current = np.mean(
                                    (
                                        y[batch_indexes]
                                        - self.__call__(
                                            X_premises[batch_indexes, :],
                                            X_consequences[batch_indexes, :],
                                        )
                                    )
                                    ** 2
                                )
                                grad = (mse_current - mse_base) / 0.001
                                self.params[param][i] -= 0.001
                                self.params[param][i] -= lr * grad

                mse_lr = np.mean((y - self.__call__(X_premises, X_consequences)) ** 2)

                if mse_lr > mse_base_lr and learning_rate == "adaptive":
                    lr = lr / 5.0

                if learning_rate == "invscaling":
                    lr = learning_rate_init / np.power(iter + 1, power_t)

                history["loss"].append(mse_lr)

        return history

    # def compute_lstsq(self, X_antecedents, X_consequents, y):

    #     current_mse = np.mean((y - self.__call__(X_antecedents, X_consequents)) ** 2)
    #     current_intercept_ = self.intercept_
    #     current_coefs_ = self.coefs_

    #     #
    #     #
    #     #
    #     NRULES = len(self.rules)
    #     NVARS = len(self.rules[0])

    #     memberships = self.rule_memberships.copy()
    #     memberships = np.repeat(memberships, NVARS, axis=1)
    #     x_ = np.tile(X_consequents, (1, NRULES))
    #     A = np.append(memberships * x_, memberships, axis=1)

    #     solution = np.linalg.lstsq(A, y, rcond=None)[0]
    #     self.intercept_ = solution[-NRULES:]

    #     solution = solution[:-NRULES]
    #     coefs = solution[:-NRULES].reshape((NRULES, NVARS))
    #     self.coefs_ = coefs

    #     mse = np.mean((y - self.__call__(X_antecedents, X_consequents)) ** 2)

    #     if mse >= current_mse:
    #         self.intercept_ = current_intercept_
    #         self.coefs_ = current_coefs_

    def plot_fuzzysets(self, i_var):
        #
        def plot_trimf():

            param_a = self.params["a"][i_var]
            param_b = self.params["b"][i_var]
            param_c = self.params["c"][i_var]

            for i_fuzzy_set in range(n_sets):

                a = param_a[i_fuzzy_set]
                b = param_b[i_fuzzy_set]
                c = param_c[i_fuzzy_set]

                membership = np.maximum(
                    0, np.minimum((x - a) / (b - a), (c - x) / (c - b))
                )
                plt.gca().plot(x, membership)

        def plot_gaussmf():

            param_a = self.params["a"][i_var]
            param_b = self.params["b"][i_var]

            for i_fuzzy_set in range(n_sets):

                a = param_a[i_fuzzy_set]
                b = param_b[i_fuzzy_set]

                membership = np.exp(-(((x - a) / b) ** 2))

                plt.gca().plot(x, membership)

        def plot_gbellmf():

            param_a = self.params["a"][i_var]
            param_b = self.params["b"][i_var]
            param_c = self.params["c"][i_var]

            for i_fuzzy_set in range(n_sets):

                a = param_a[i_fuzzy_set]
                b = param_b[i_fuzzy_set]
                c = param_c[i_fuzzy_set]

                membership = 1 / (1 + np.power(((x - a) / b) ** 2, c))
                plt.gca().plot(x, membership)

        n_sets = self.num_input_mfs[i_var]
        x_min = self.x_min[i_var]
        x_max = self.x_max[i_var]
        x = np.linspace(start=x_min, stop=x_max, num=100)

        plot_fn = {
            "trimf": plot_trimf,
            "gaussmf": plot_gaussmf,
            "gbellmf": plot_gbellmf,
        }[self.mftype]

        plot_fn()
        plt.gca().set_ylim(-0.05, 1.05)
        plt.gca().spines["left"].set_color("lightgray")
        plt.gca().spines["bottom"].set_color("gray")
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)


# rng = np.random.default_rng(12345)
# x1 = np.linspace(start=0, stop=10, num=100)
# x2 = rng.uniform(0, 10, 100)
# y1 = np.sin(x1) + np.cos(x1)
# y2 = (y1) / np.exp(x1)

# import matplotlib.pyplot as plt
# import pandas as pd


# X = pd.DataFrame({"x1": x1, "x2": x2})

# m = Sugeno(num_input_mfs=(3, 3), mftype="gbellmf", seed=1234567)

# # m(X.values, X.values)

# history = m.fit(
#     X.values,
#     X.values,
#     y2,
#     max_iter=300,
#     learning_rate="invscaling",
#     learning_rate_init=0.3,
#     power_t=0.5,
#     batch_size=20,
#     shuffle=True,
# )

# plt.figure(figsize=(10, 6))
# plt.subplot(1, 2, 1)
# plt.plot(history["loss"])
# print("\n", np.mean((y2 - m(X.values, X.values)) ** 2))

# # m.fit(X.values, X.values, y2, learning_rate=0.1, max_iter=200)

# # m.plot_fuzzysets(0)
# # m.plot_fuzzysets(1)
# # np.mean((y2 - m(X.values, X.values)) ** 2)

# # m(X.values)

# plt.subplot(1, 2, 2)
# y_pred = m(X.values, X.values)
# plt.plot(y2, "-k")
# plt.plot(y_pred, "-r")
