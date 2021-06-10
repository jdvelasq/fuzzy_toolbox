"""
Fuzzy Variables
==============================================================================

"""

import matplotlib.pyplot as plt
import numpy as np


class FuzzyVariable:
    """Creates a (fuzzy) linguistic variable.

    Args:
        name (string): variable name.
        universe (list, numpy.array): list of points defining the universe of the variable.
        sets (dict): dictionary where keys are the name of the sets, and the values correspond to the membership for each point of the universe.

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

    def __getitem__(self, name):
        """Returns the membership function for the specified fuzzy set.

        Args:
            name (string): Fuzzy set name

        Returns:
            A numpy array.

        """
        return self.sets[name]

    def __setitem__(self, name, memberships):
        """Sets the membership function values for the specified fuzzy set.

        Args:
            name (string): Fuzzy set name.
            memberships (list, numpy.array): membership values.

        """
        self.sets[name] = np.array(memberships)

    def plot(self, figsize=(10, 3)):
        """Plots the fuzzy sets defined for the variable.

        Args:
            figsize (tuple): figure size.

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

    def get_modified_membership(self, fuzzyset, modifiers):
        """Computes a modified membership function.

        Args:
            fuzzyset (string): Identifier of the fuzzy set.
            modifier (string): {"very"|"somewhat"|"more_or_less"|"extremely"|"plus"|"intensify"|"slightly"|None}
            negation (bool): When True computes the negation of the fuzzy set.

        Returns:
            A numpy.array.

        """
        if modifiers is None:
            return self.sets[fuzzyset]

        membership = self.sets[fuzzyset]
        for modifier in modifiers:

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

            if modifier.upper() == "NOT":
                membership = 1 - membership

        return membership

    def compute_membership(self, value, fuzzyset, modifiers=None):
        """Computes the value of the membership function on a specifyied point of the universe for the fuzzy set.

        Args:
            value (float, numpy.array): point to evaluate the value of the membership function.
            fuzzyset (string): name of the fuzzy set.
            modifier (string): membership function modifier.
            negation (bool): returns the negation?.

        Returns:
            A float number or numpy.array.
        """

        membership = self.get_modified_membership(fuzzyset, modifiers)

        return np.interp(
            x=value,
            xp=self.universe,
            fp=membership,
        )

    def aggregate(self, operator="max"):
        """Replace the fuzzy sets by a unique fuzzy set computed by the aggregation operator.

        Args:
            operator (string): {"max"|"sim"|"probor"} aggregation operator.

        Returns:
            A FuzzyVariable

        """
        aggregation = None
        result = FuzzyVariable(name=self.name, universe=self.universe)

        for i_fuzzyset, fuzzyset in enumerate(self.sets.keys()):

            result[fuzzyset] = self.sets[fuzzyset]

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

        result["_aggregation_"] = aggregation
        return result

    def defuzzificate(self, fuzzyset="_aggregation_", operator="cog"):
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
            membership = self.compute_membership(x, fuzzyset)
            return np.sum(x * membership) / sum(membership)

        if operator == "bisection":
            start = np.min(self.universe)
            stop = np.max(self.universe)
            x = np.linspace(start, stop, num=200)
            membership = self.compute_membership(x, fuzzyset)
            area = np.sum(membership) / 2
            for i in range(len(x)):
                if np.sum(membership[0:i]) > area:
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
