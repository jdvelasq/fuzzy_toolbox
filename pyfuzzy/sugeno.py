import numpy as np
import progressbar
import logging


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

    def compute_memberships_by_rule(self, X):

        NPOINTS = len(X)
        NRULES = len(self.rules)
        NDIM = len(self.rules[0])

        fuzzy_index = np.tile(self.rules, (NPOINTS, 1))

        data = np.repeat(X, NRULES, axis=0)
        rule_memberships = np.zeros(shape=(NPOINTS * NRULES, NDIM))

        for i_dim in range(NDIM):

            fuzzy_sets = self.fuzzy_sets[i_dim]
            fuzzy_sets = fuzzy_sets[fuzzy_index[:, i_dim]]

            x = data[:, i_dim]
            a = fuzzy_sets[:, 0]
            b = fuzzy_sets[:, 1]
            c = fuzzy_sets[:, 2]

            membership = np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))
            rule_memberships[:, i_dim] = membership

        rule_memberships = rule_memberships.min(axis=1)
        self.rule_memberships = rule_memberships.reshape((NPOINTS, NRULES))

    def compute_consequents(self, X):

        NRULES = len(self.rules)
        NPOINTS = len(X)

        output = np.matmul(X, np.transpose(self.coefs_))
        intercepts = self.intercept_.reshape((1, NRULES))
        intercepts = np.tile(intercepts, (NPOINTS, 1))
        self.consequents = output + intercepts

    def compute_weighted_output(self):

        NRULES = len(self.rules)
        NPOINTS = len(X)

        sum_of_memberships = self.rule_memberships.sum(axis=1).reshape((NPOINTS, 1))
        sum_of_memberships = np.tile(sum_of_memberships, (1, NRULES))
        rule_memberships = self.rule_memberships / sum_of_memberships
        output = rule_memberships * self.consequents
        return output.sum(axis=1).reshape(-1)

    def __call__(self, X):

        self.compute_memberships_by_rule(X)
        self.compute_consequents(X)
        return self.compute_weighted_output()

    def fit(self, X, y, learning_rate=0.01, max_iter=10):

        self.create_structure(X)

        for _ in progressbar.progressbar(range(max_iter)):

            self.improve_fuzzysets(X, y, learning_rate)
            self.improve_coefs(X, y, learning_rate)
            self.improve_intercepts(X, y, learning_rate)

        print("Final MSE = {:5.3f}".format(np.mean((y - self.__call__(X)) ** 2)))

    def improve_fuzzysets(self, X, y, learning_rate):

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

            self.fuzzy_sets[i_var] = self.fuzzy_sets[i_var] - learning_rate * grad

    def improve_intercepts(self, X, y, learning_rate):

        y_pred = self.__call__(X)
        mse_base = np.mean((y - y_pred) ** 2)

        grad = np.zeros(shape=self.intercept_.shape)

        for i_row in range(self.intercept_.shape[0]):

            self.intercept_[i_row] += 0.001
            y_pred = self.__call__(X)
            mse_current = np.mean((y - y_pred) ** 2)
            grad[i_row] = (mse_current - mse_base) / 0.001
            self.intercept_[i_row] -= 0.001

        self.intercept_ = self.intercept_ - learning_rate * grad

    def improve_coefs(self, X, y, learning_rate):

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

        self.coefs_ = self.coefs_ - learning_rate * grad

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
m.fit(X.values, y2, learning_rate=0.01, max_iter=500)
np.mean((y2 - m(X.values)) ** 2)

# m(X.values)

y_pred = m(X.values)
plt.plot(y2, "-k")
plt.plot(y_pred, "-r")