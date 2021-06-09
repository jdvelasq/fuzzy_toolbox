"""
Fuzzy Variables
==============================================================================

"""


import matplotlib.pyplot as plt
import numpy as np


def gaussmf(x, center=0, sigma=1):
    """Gaussian membership function.

    This function computes fuzzy membership values using a Gaussian membership function using NumPy.

    Args:
        x (float, np.array): input value.
        center (float): Center of the distribution.
        sigma (float): standard deviation.

    Returns:
        A numpy.array.
    """
    return np.exp(-((x - center) ** 2) / (2 * sigma))


def gbellmf(x, a=1, b=1, c=0):
    """Generalized bell-shaped membership function.

    This function computes fuzzy membership values using a generalized bell membership function using NumPy.

    Args:
        x (float, np.array): input value.
        a (float): standard deviation.
        b (float): exponent.
        c (float): center.

    Returns:
        A numpy.array.
    """
    return 1 / (1 + np.abs((x - c) / a) ** (2 * b))


def trimf(x, a, b, c):
    """Triangular membership function.

    This function computes fuzzy membership values using a triangular membership function using NumPy.

    Args:
        x (float, np.array): input value.
        a (float): Left feet.
        b (float): center or peak.
        c (float): right feet.

    Returns:
        A numpy.array.
    """
    return np.where(
        x <= a,
        0,
        np.where(x <= b, (x - a) / (b - a), np.where(x <= c, (c - x) / (c - b), 0)),
    )


def pimf(x, a, b, c, d):
    """Pi-shaped membership function.

    This function computes fuzzy membership values using a pi-shaped membership function using NumPy.

    Args:
        x (float, np.array): input value.
        a (float): Left feet.
        b (float): Left peak.
        c (float): Right peak.
        d (float): Right feet.

    Returns:
        A numpy.array.
    """
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
    """Sigmoidal membership function.

    This function computes fuzzy membership values using a sigmoidal membership function using NumPy.

    Args:
        x (float, np.array): input value.
        a (float): slope.
        c (float): center.

    Returns:
        A numpy.array.
    """
    return 1 / (1 + np.exp(-a * (x - c)))


def smf(x, a, b):
    """S-shaped membership function

    This function computes fuzzy membership values using a S-shaped membership function using NumPy.

    Args:
        x (float, np.array): input value.
        a (float): Left feet.
        b (float): Right peak.

    Returns:
        A numpy.array.
    """
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
    """Trapezoida membership function

    This function computes fuzzy membership values using a trapezoidal membership function using NumPy.

    Args:
        x (float, np.array): input value.
        a (float): Left feet.
        b (float): Left peak.
        c (float): Right peak.
        d (float): Right feet.

    Returns:
        A numpy.array.
    """

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
    """Z-shaped membership function

    This function computes fuzzy membership values using a Z-shaped membership function using NumPy.

    Args:
        x (float, np.array): input value.
        a (float): Left peak.
        b (float): Right feet.

    Returns:
        A numpy.array.
    """
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

    def __init__(self, name, universe, sets=None):
        self.name = name.replace(" ", "_")
        self.universe = universe
        if sets is None:
            self.sets = {}
        else:
            self.sets = sets
            for key in sets.keys():
                self.sets[key] = np.array(self.sets[key])

    def __getitem__(self, key):
        return self.sets[key]

    def __setitem__(self, key, value):
        self.sets[key] = np.array(value)

    def plot(self, figsize=(10, 3)):
        """Plots the fuzzy sets defined for the variable.

        Args:
            figsize: figure size.

        Returns:
            Nothing.


        """
        plt.figure(figsize=figsize)
        for k in self.sets.keys():
            plt.plot(self.universe, self.sets[k], "o-", label=k)
        plt.legend()
        plt.title(self.name)
        plt.ylim(-0.05, 1.05)
        plt.gca().spines["left"].set_color("lightgray")
        plt.gca().spines["bottom"].set_color("gray")
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

    def get_modified_membership(self, fuzzyset, modifier, negation):
        """Computes a modified membership function.

        Args:
            fuzzyset (string): Identifier of the fuzzy set.
            modifier (string): {"very"|"somewhat"|"more_or_less"|"extremely"|"plus"|"intensify"|"slightly"|None}
            negation (bool): When True computes the negation of the fuzzy set.

        """

        membership = self.sets[fuzzyset]
        if modifier is not None:
            if modifier.upper() == "VERY":
                membership = membership ** 2
            if modifier.upper() == "SOMEWHAT":
                membership = membership ** 0.33333
            if modifier.upper().replace("-", "_") == "MORE_OR_LESS":
                membership = membership ** 0.5
            if modifier.upper() == "EXTREMELY":
                membership = membership ** 3
            if modifier.upper() == "PLUS":
                membership = membership ** 1.25
            if modifier.upper() == "INTENSIFY":
                membership = np.where(
                    membership <= 0.5, membership ** 2, 1 - 2 * (1 - membership) ** 2
                )
            if modifier.upper() == "SLIGHTLY":
                plus_membership = membership ** 1.25
                not_very_membership = 1 - membership ** 2
                membership = np.where(
                    plus_membership < not_very_membership,
                    plus_membership,
                    not_very_membership,
                )
                membership = membership / np.max(membership)
                membership = np.where(
                    membership <= 0.5, membership ** 2, 1 - 2 * (1 - membership) ** 2
                )

        if negation is True:
            membership = 1 - membership

        return membership

    def membership(self, value, fuzzyset, modifier=None, negation=False):
        """Computes the valor of the membership function on a specifyied point of the universe for the fuzzy set.

        Args:
            value (float): point to evaluate the value of the membership function.
            fuzzyset (string): name of the fuzzy set.
            modifier (string): membership function modifier
            negation (bool): returns the negation?

        Returns:
            A float number.
        """

        membership = self.get_modified_membership(fuzzyset, modifier, negation)

        return np.interp(
            x=value,
            xp=self.universe,
            fp=membership,
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
        """Computes a representative crisp value for the fuzzy set.

        Args:
            fuzzyset (string): Fuzzy set to defuzzify
            operator (string): {"cog"|"bisection"|"mom"|"lom"|"som"}

        Returns:
            A float value.

        """

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
