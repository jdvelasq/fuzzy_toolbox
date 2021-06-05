import matplotlib.pyplot as plt
import numpy as np


def gaussmf(x, center=0, sigma=1):
    """ """
    return np.exp(-((x - center) ** 2) / (2 * sigma))


def gbellmf(x, a=1, b=1, c=0):
    """ """
    return 1 / (1 + np.abs((x - c) / a) ** (2 * b))


def trimf(x, a, b, c):
    """ """
    return np.where(
        x <= a,
        0,
        np.where(x <= b, (x - a) / (b - a), np.where(x <= c, (c - x) / (c - b), 0)),
    )


def pimf(x, a, b, c, d):
    return np.where(
        x <= a,
        0,
        np.where(
            x <= (a + b) / 2.0,
            2 * ((x - a) / (b - a)) ** 2,
            np.where(
                x <= c,
                1,
                np.where(
                    x <= (c + d) / 2.0,
                    1 - 2 * ((x - c) / (d - c)) ** 2,
                    np.where(x <= d, 2 * ((x - d) / (d - c)) ** 2, 0),
                ),
            ),
        ),
    )


def sigmf(x, a, c):
    return 1 / (1 + np.exp(-a * (x - c)))


def smf(x, a, b):
    return np.where(
        x <= a,
        0,
        np.where(
            x <= (a + b) / 2,
            2 * ((x - a) / (b - a)) ** 2,
            np.where(x <= b, 1 - 2 * ((x - b) / (b - a)) ** 2, 1),
        ),
    )


def trapmf(x, a, b, c, d):
    return np.where(
        x <= a,
        0,
        np.where(
            x <= b,
            (x - a) / (b - a),
            np.where(x <= c, 1, np.where(x <= d, (d - x) / (d - c), 0)),
        ),
    )


def zmf(x, a, b):
    return np.where(
        x <= a,
        1,
        np.where(
            x <= (a + b) / 2,
            1 - 2 * ((x - a) / (b - a)) ** 2,
            np.where(x <= b, 2 * ((x - b) / (b - a)) ** 2, 0),
        ),
    )


class FuzzyVariable:
    """Creates a linguistic variable.

    Args:
        name (string): variable name.
        universe (list, numpy.array): list of points defining the universe of the variable.
        sets (dict): dictioary where keys are the name of the sets, and the values correspond to the membership for each point of the universe.

    Returns:
        A fuzzy variable.

    """

    def __init__(self, name, universe, sets={}):
        self.name = name
        self.universe = universe
        self.sets = sets

    def __getitem__(self, key):
        return self.sets[key]

    def __setitem__(self, key, value):
        self.sets[key] = value

    def plot(self, figsize=(10, 3)):
        """Plots the fuzzy sets defined for the variable.

        Args:
            figsize: figure size.

        Returns:
            Nothing.


        """
        # plt.gcf().clf()
        plt.close()
        plt.figure(figsize=figsize)
        plt.cla()
        for k in self.sets.keys():
            plt.plot(self.universe, self.sets[k], "o-", label=k)
        plt.legend()
        plt.title(self.name)
        plt.ylim(-0.05, 1.05)
        plt.gca().spines["left"].set_color("lightgray")
        plt.gca().spines["bottom"].set_color("gray")
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        # plt.show()
        # plt.close("all")

    def membership(self, value, fuzzyset):
        """Computes the valor of the membership function on a specifyied point of the universe for the fuzzy set.

        Args:
            value (float): point to evaluate the value of the membership function.
            fuzzyset (string): name of the fuzzy set.

        Returns:
            A float number.
        """  #
        return np.interp(
            x=value,
            xp=self.universe,
            fp=self.sets[fuzzyset],
        )

    def aggregate(self, operator="max"):
        """Transforms the fuzzy sets to a unique fuzzy set computed by the aggregation operator.

        Args:
            operator (string): {"max"|"sim"|"probor"}

        Returns:
            Nothing

        """
        aggregation = None

        for i_fuzzyset, fuzzyset in enumerate(self.sets.keys()):

            if i_fuzzyset == 0:
                aggregation = self.sets[fuzzyset]
            else:
                if operator == "max":
                    aggregation = np.maximum(aggregation, self.sets[fuzzyset])
                if operator == "sum":
                    aggregation = aggregation + self.sets[fuzzyset]
                if operator == "probor":
                    aggregation = (
                        aggregation
                        + self.sets[fuzzyset]
                        - aggregation * self.sets[fuzzyset]
                    )

        self.sets = {"aggregation": aggregation}

    def defuzzification(self, fuzzyset="aggregation", operator="cog"):
        """ """

        if operator == "cog":
            #
            # cog: center of gravity
            #
            start = np.min(self.universe)
            stop = np.max(self.universe)
            x = np.linspace(start, stop, num=200)
            memberships = self.membership(x, fuzzyset)
            return np.sum(x * memberships) / sum(memberships)

        if operator == "bisection":
            start = np.min(self.universe)
            stop = np.max(self.universe)
            x = np.linspace(start, stop, num=200)
            memberships = self.membership(x, fuzzyset)
            area = np.sum(memberships) / 2
            for i in range(len(x)):
                if np.sum(memberships[0:i]) > area:
                    return x[i]
            return -1

        maximum = np.max(self.sets[fuzzyset])
        maximum = np.array(
            [x for x, m in zip(self.universe, self.sets[fuzzyset]) if m == maximum]
        )

        if operator == "mom":
            #
            # MoM: Mean of the values for which the output fuzzy set is maximum
            #
            return np.mean(maximum)

        if operator == "lom":
            #
            # lom: Largest value for which the output fuzzy set is maximum
            #
            return np.max(maximum)

        if operator == "som":
            #
            # som: Smallest value for which the output fuzzy set is maximum
            #
            return np.min(maximum)
