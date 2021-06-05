import numpy as np


class Sugeno:
    def __init__(self, num_input_mfs=(3,), mftype="trimf", seed=None):
        self.num_input_mfs = num_input_mfs
        self.mftype = mftype

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

        self.antecedents = None
        self.rule_output = None

    def __call__(self, X):

        # computa la pertenencia por cada componente del antecedente
        # de cada regla
        self.compute_memberships(X)
        self.reduce_memberships()
        self.compute_consequents(X)
        self.computed_weighted_output()
        return self.output

    def computed_weighted_output(self):

        sum_of_memberships = np.sum(self.reduced_memberships, axis=1)

        for i_row in range(len(self.reduced_memberships)):
            self.reduced_memberships[i_row, :] /= sum_of_memberships[i_row]

        self.output = self.rule_output * self.reduced_memberships
        self.output = self.output.sum(axis=1)

    def compute_consequents(self, X):

        self.rule_output = np.zeros(shape=(len(X), len(self.rules)))
        for i_row, xi in enumerate(X):
            self.rule_output[i_row, :] = np.matmul(self.coefs_, xi) + self.intercept_

    def reduce_memberships(self):

        self.reduced_memberships = np.zeros(
            shape=(len(self.antecedents), len(self.rules))
        )
        for i_row in range(len(self.antecedents)):
            for i_rule in range(len(self.rules)):
                self.reduced_memberships[i_row, i_rule] = np.min(
                    self.antecedents[i_row, i_rule, :]
                )

    def compute_memberships(self, X):
        def compute_membership_per_set(var, fuzzy_sets):

            a = fuzzy_sets[:, 0]
            b = fuzzy_sets[:, 1]
            c = fuzzy_sets[:, 2]

            return np.where(
                var <= a,
                0,
                np.where(
                    var <= b,
                    (var - a) / (b - a),
                    np.where(var <= c, (c - var) / (c - b), 0),
                ),
            )

        self.antecedents = np.zeros(shape=(len(X), len(self.rules), len(self.rules[0])))

        for i in range(self.antecedents.shape[0]):
            self.antecedents[i, :, :] = np.array(self.rules)

        for i_row, x_row in enumerate(X):

            for i_var, var in enumerate(x_row):
                fuzzy_sets = self.fuzzy_sets[i_var]
                m = compute_membership_per_set(var, fuzzy_sets)

                for i_rule in range(len(self.rules)):
                    i_set = self.antecedents[i_row, i_rule, i_var]
                    self.antecedents[i_row, i_rule, i_var] = m[int(i_set)]

    def fit(self, X, y):

        self.create_structure(X)

        # self.improve_fuzzysets(X, y)

        for _ in range(100):

            self.improve_fuzzysets(X, y)
            self.improve_coefs(X, y)
            self.improve_intercepts(X, y)

            print(np.mean((y - self.__call__(X)) ** 2))

    def improve_fuzzysets(self, X, y):

        y_pred = self.__call__(X)
        mse_base = np.mean((y - y_pred) ** 2)

        for i_var in range(len(self.fuzzy_sets)):

            grad = np.zeros(shape=self.fuzzy_sets[i_var].shape)

            for i_set in range(len(self.fuzzy_sets[i_var])):
                for i_comp in range(3):

                    self.fuzzy_sets[i_var][i_set, i_comp] += 0.001

                    y_pred = self.__call__(X)
                    mse_current = np.mean((y - y_pred) ** 2)
                    grad[i_set, i_comp] = (mse_current - mse_base) / 0.001

                    self.fuzzy_sets[i_var][i_set, i_comp] -= 0.001

            self.fuzzy_sets[i_var] = self.fuzzy_sets[i_var] - 0.001 * grad

    def improve_intercepts(self, X, y):

        y_pred = self.__call__(X)
        mse_base = np.mean((y - y_pred) ** 2)

        grad = np.zeros(shape=self.intercept_.shape)

        for i_row in range(self.intercept_.shape[0]):

            self.intercept_[i_row] += 0.001
            y_pred = self.__call__(X)
            mse_current = np.mean((y - y_pred) ** 2)
            grad[i_row] = (mse_current - mse_base) / 0.001
            self.intercept_[i_row] -= 0.001

        self.intercept_ = self.intercept_ - 0.001 * grad

    def improve_coefs(self, X, y):

        y_pred = self.__call__(X)
        mse_base = np.mean((y - y_pred) ** 2)

        grad = np.zeros(shape=self.coefs_.shape)

        for i_row in range(self.coefs_.shape[0]):
            for i_col in range(self.coefs_.shape[1]):

                self.coefs_[i_row, i_col] += 0.001

                y_pred = self.__call__(X)
                mse_current = np.mean((y - y_pred) ** 2)

                grad[i_row, i_col] = (mse_current - mse_base) / 0.001

                self.coefs_[i_row, i_col] -= 0.001

        self.coefs_ = self.coefs_ - 0.001 * grad

    def create_structure(self, X):

        self.create_rules()
        self.create_antecedents(X)
        self.create_consequents()

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

    def create_antecedents(self, X):

        x_min = X.min(axis=0)
        x_max = X.max(axis=0)

        antecedents = []
        for i_var in range(len(x_min)):

            n_divs = self.num_input_mfs[i_var] - 1
            n_sets = self.num_input_mfs[i_var]

            delta_x = (x_max[i_var] - x_min[i_var]) / n_divs
            matrix = np.zeros(shape=(n_sets, 3))

            matrix[:, 0] = np.linspace(
                start=x_min[i_var] - delta_x, stop=x_max[i_var] - delta_x, num=n_sets
            )
            matrix[:, 1] = np.linspace(
                start=x_min[i_var], stop=x_max[i_var], num=n_sets
            )
            matrix[:, 2] = np.linspace(
                start=x_min[i_var] + delta_x, stop=x_max[i_var] + delta_x, num=n_sets
            )

            antecedents.append(matrix)

        # self.fuzzy_sets = np.array(antecedents, dtype=np.int0)
        self.fuzzy_sets = antecedents

    def create_consequents(self):
        n_vars = len(self.num_input_mfs)
        n_rules = np.prod(self.num_input_mfs)
        self.coefs_ = self.rng.normal(loc=0, scale=0.1, size=(n_rules, n_vars))
        self.intercept_ = self.rng.normal(loc=0, scale=0.1, size=n_rules)


x1 = np.linspace(start=0, stop=10, num=100)
x2 = np.random.uniform(0, 10, 100)
y1 = np.sin(x1) + np.cos(x1)
y2 = y1 / np.exp(x1)

import matplotlib.pyplot as plt
import pandas as pd

X = pd.DataFrame({"x1": x1, "x2": x2})

m = Sugeno(num_input_mfs=(3, 4))

m.fit(X.values, y2)

y_pred = m(X.values)

plt.plot(y2, "-k")
plt.plot(y_pred, "-r")
