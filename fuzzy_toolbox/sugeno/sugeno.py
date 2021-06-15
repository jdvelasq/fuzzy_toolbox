"""
Mamdani fuzzy model
==============================================================================

"""
import matplotlib.pyplot as plt
import numpy as np


class Sugeno:
    """Crates a Sugeno inference system.

    Args:
        rules (list): List of fuzzy rules.
        and_operator (string): {"min", "prod"}. Operator used for rules using AND.
        or_operator (string): {"max"|"probor"}. Operator used for rules using OR.

    """

    def __init__(
        self,
        rules,
        and_operator="min",
        or_operator="max",
    ):
        self.rules = rules
        self.and_operator = and_operator
        self.or_operator = or_operator
        #
        self.output = None
        self.values = None

    def __call__(self, **values):
        """Computes the output of the Mamdani inference system.

        Args:
            values (dict): Values for the variables in the antecedent.
        """
        self.values = values
        self.compute_rules()
        self.compute_aggregation()
        return self.output

    def compute_rules(self):
        """Computes the output fuzzy set of each rule."""
        for rule in self.rules:
            rule.compute_rule(
                and_operator=self.and_operator,
                or_operator=self.or_operator,
                **self.values,
            )

    def compute_aggregation(self):
        """Computes the output crisp value of the inference system."""

        firing_strengths = [rule.firing_strength for rule in self.rules]
        sum_firing_strengths = sum(firing_strengths)
        norm_firing_strengths = [nfs / sum_firing_strengths for nfs in firing_strengths]

        consequences = [rule.consequence for rule in self.rules]

        self.output = np.sum(
            [
                nfs * consequence
                for nfs, consequence in zip(norm_firing_strengths, consequences)
            ]
        )

    def plot(self, **values):
        n_rows = len(self.rules) + 1

        n_antecedents = max([len(rule.antecedents) for rule in self.rules])

        names = []
        for rule in self.rules:
            for antecedent in rule.antecedents:
                names.append(antecedent[0].name)

        names = sorted(set(names))
        positions = {n: i for i, n in enumerate(names)}

        #
        # Plots rules
        #
        for i_rule, rule in enumerate(self.rules):
            rule.plot(
                n_rows=n_rows,
                i_row=i_rule,
                and_operator=self.and_operator,
                or_operator=self.or_operator,
                n_antecedents=n_antecedents,
                positions=positions,
                **values,
            )

        #
        # Delete plot titles
        #
        for i_rule, rule in enumerate(self.rules):
            for i, _ in enumerate(rule.antecedents):
                if i_rule != 0:
                    plt.subplot(
                        n_rows, n_antecedents + 1, i_rule * (n_antecedents + 1) + i + 1
                    )
                    plt.gca().set_title("")

            # plt.subplot(
            #     n_rows,
            #     n_antecedents + 1,
            #     i_rule * (n_antecedents + 1) + n_antecedents + 1,
            # )
            # plt.gca().set_title("")

        #
        # Remove xaxis
        #
        for i_rule, rule in enumerate(self.rules):
            for i, _ in enumerate(rule.antecedents):
                plt.gca().spines["right"].set_visible(False)
                plt.gca().spines["left"].set_visible(False)
                plt.gca().spines["top"].set_visible(False)

                if i_rule + 1 < len(self.rules):
                    plt.subplot(
                        n_rows, n_antecedents + 1, i_rule * (n_antecedents + 1) + i + 1
                    )
                    plt.gca().get_xaxis().set_visible(False)

        #     plt.subplot(
        #         n_rows,
        #         n_antecedents + 1,
        #         i_rule * (n_antecedents + 1) + n_antecedents + 1,
        #     )
        #     plt.gca().get_xaxis().set_visible(False)

        #
        # System output
        #

        result = self.__call__(**values)

        plt.subplot(n_rows, n_antecedents + 1, n_rows * (n_antecedents + 1))

        text_kwargs = dict(ha="center", va="center", fontsize=18, color="black")
        plt.gca().text(0.5, 0.5, str(round(result, 2)), **text_kwargs)
        plt.gca().spines["left"].set_visible(True)
        plt.gca().spines["bottom"].set_visible(True)
        plt.gca().spines["top"].set_visible(True)
        plt.gca().spines["right"].set_visible(True)
        plt.gca().get_yaxis().set_visible(False)
        plt.gca().get_xaxis().set_visible(False)


# if __name__ == "__main__":

#     #
#     # Takagy y Sugeno
#     #

#     from rule import SugenoRule

#     import sys

#     sys.path.insert(0, "..")

# from mamdani.variable import FuzzyVariable
# from mamdani.mf import *

# x = np.linspace(start=0, stop=20, num=200)
# x1 = FuzzyVariable(
#     name="x1",
#     universe=x,
#     sets={
#         "small_1": trimf(x, 0, 0, 16),
#         "big_1": trimf(x, 10, 20, 20),
#     },
# )

# x = np.linspace(start=0, stop=10, num=100)
# x2 = FuzzyVariable(
#     name="x2",
#     universe=x,
#     sets={
#         "small_2": trimf(x, 0, 0, 8),
#         "big_2": trimf(x, 2, 10, 20),
#     },
# )

# rule_1 = SugenoRule(
#     antecedents=[
#         (x1, "small_1"),
#         (x2, "small_2"),
#     ],
#     consequent=lambda x1, x2: x1 + x2,
# )

# rule_2 = SugenoRule(
#     antecedents=[
#         (x1, "big_1"),
#     ],
#     consequent=lambda x1, x2: 2.0 * x1,
# )

# rule_3 = SugenoRule(
#     antecedents=[
#         (x2, "big_2"),
#     ],
#     consequent=lambda x1, x2: 3.0 * x2,
# )

# sugeno = Sugeno(
#     rules=[rule_1, rule_2, rule_3],
#     and_operator="min",
#     or_operator="max",
# )

#     print(sugeno(x1=12, x2=5))

#     plt.figure(figsize=(12, 8))

#     sugeno.plot(x1=12, x2=5)
