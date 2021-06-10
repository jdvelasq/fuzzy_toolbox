"""
Mamdani fuzzy model
==============================================================================

"""
import matplotlib.pyplot as plt
import numpy as np

from .rule import FuzzyRule
from .variable import FuzzyVariable

#
#
# Membership functions
#
#


class Mamdani:
    """Crates a Mamdani inference system.

    Args:
        rules (list): List of fuzzy rules.
        and_operator (string): {"min", "prod"}. Operator used for rules using AND.
        or_operator (string): {"max"|"probor"}. Operator used for rules using OR.
        implication_operator (string): {"min", "prod"}.
        aggregation_operator (string): {"max"|"probor"}.
        defuzzification_operator (string): {"cog"|"bisection"|"mom"|"lom"|"som"}


    """

    def __init__(
        self,
        rules,
        and_operator="min",
        or_operator="max",
        implication_operator="min",
        aggregation_operator="max",
        defuzzification_operator="cog",
    ):
        self.rules = rules
        self.and_operator = and_operator
        self.or_operator = or_operator
        self.implication_operator = implication_operator
        self.aggregation_operator = aggregation_operator
        self.defuzzification_operator = defuzzification_operator
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
        return self.compute_defuzzification()

    def compute_rules(self):
        """Computes the output fuzzy set of each rule."""
        for rule in self.rules:
            rule.compute_inference(
                and_operator=self.and_operator,
                or_operator=self.or_operator,
                implication_operator=self.implication_operator,
                **self.values,
            )

    def compute_aggregation(self):
        """Computes the output fuzzy set of the inference system."""
        for i_rule, rule in enumerate(self.rules):
            if i_rule == 0:
                self.output = FuzzyVariable(
                    name="output",
                    universe=rule.output.universe,
                    sets={"rule_{}".format(i_rule): rule.output.sets["rule_output"]},
                )
            else:
                self.output.sets["rule_{}".format(i_rule)] = rule.output.sets[
                    "rule_output"
                ]

        #
        # Agrega los conjuntos borrosos para obtener uno
        # solo (el código aparece en la definición de la
        # clase)
        #
        self.output = self.output.aggregate()

    def compute_defuzzification(self):
        """Computes the equivalent crisp value representing the output fuzzy set of the system."""
        return self.output.defuzzificate(operator=self.defuzzification_operator)

    def plot(self, **values):
        n_rows = len(self.rules) + 1

        n_antecedents = max([len(rule.antecedents) for rule in self.rules])

        #
        # Plots rules
        #
        for i_rule, rule in enumerate(self.rules):
            rule.plot(
                n_rows=n_rows,
                i_row=i_rule,
                and_operator=self.and_operator,
                or_operator=self.or_operator,
                implication_operator=self.implication_operator,
                n_antecedents=n_antecedents,
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

            plt.subplot(
                n_rows,
                n_antecedents + 1,
                i_rule * (n_antecedents + 1) + n_antecedents + 1,
            )
            plt.gca().set_title("")

        #
        # Remove xaxis
        #
        for i_rule, rule in enumerate(self.rules):
            for i, _ in enumerate(rule.antecedents):
                if i_rule + 1 < len(self.rules):
                    plt.subplot(
                        n_rows, n_antecedents + 1, i_rule * (n_antecedents + 1) + i + 1
                    )
                    plt.gca().get_xaxis().set_visible(False)

            plt.subplot(
                n_rows,
                n_antecedents + 1,
                i_rule * (n_antecedents + 1) + n_antecedents + 1,
            )
            plt.gca().get_xaxis().set_visible(False)

        #
        # System output
        #

        result = self.__call__(**values)

        plt.subplot(n_rows, n_antecedents + 1, n_rows * (n_antecedents + 1))

        for fuzzyset in self.output.sets.keys():
            if fuzzyset == "_aggregation_":
                linewidth = 2
                color = "k"
            else:
                linewidth = 1
                color = "gray"

            plt.gca().plot(
                self.output.universe,
                self.output[fuzzyset],
                "-",
                linewidth=linewidth,
                color=color,
            )

        plt.gca().fill_between(
            self.output.universe, self.output["_aggregation_"], color="gray", alpha=0.7
        )

        plt.gca().set_ylim(-0.05, 1.05)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(True)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().get_yaxis().set_visible(True)
        plt.gca().get_xaxis().set_visible(True)
        plt.gca().yaxis.tick_right()

        plt.gca().vlines(x=result, ymin=-0.0, ymax=1.0, color="red", linewidth=2)

        plt.subplot(n_rows, n_antecedents + 1, n_antecedents + 1)
        plt.gca().set_title(
            "{} = {:0.2f}".format(self.rules[0].get_consequent_name(), result)
        )


#
# Test: Credit Decision Problem
#
# score = FuzzyVariable(
#     name="score",
#     universe=np.arange(start=150, stop=201, step=5),
#     sets={
#         "High": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.7, 1.0, 1.0, 1.0],
#         "Low": [1.0, 1.0, 0.8, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     },
# )

# ratio = FuzzyVariable(
#     name="ratio",
#     universe=[0.1, 0.3, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.5, 0.7, 1.0],
#     sets={
#         "Goodr": [1, 1, 0.7, 0.3, 0, 0, 0, 0, 0, 0, 0],
#         "Badr": [0, 0, 0, 0, 0, 0, 0, 0.3, 0.7, 1.0, 1.0],
#     },
# )

# credit = FuzzyVariable(
#     name="credit",
#     universe=list(range(11)),
#     sets={
#         "Goodc": [1, 1, 1, 0.7, 0.3, 0, 0, 0, 0, 0, 0],
#         "Badc": [0, 0, 0, 0, 0, 0, 0.3, 0.7, 1, 1, 1],
#     },
# )

# decision = FuzzyVariable(
#     name="decision",
#     universe=list(range(11)),
#     sets={
#         "Approve": [0, 0, 0, 0, 0, 0, 0.3, 0.7, 1, 1, 1],
#         "Reject": [1, 1, 1, 0.7, 0.3, 0, 0, 0, 0, 0, 0],
#     },
# )

# rule_1 = FuzzyRule(
#     antecedents=[
#         (score, "High"),
#         (ratio, "Goodr"),
#         (credit, "Goodc"),
#     ],
#     consequent=(decision, "Approve"),
# )

# rule_2 = FuzzyRule(
#     antecedents=[
#         (score, "Low"),
#         (ratio, "Badr"),
#         (credit, "Badc"),
#     ],
#     consequent=(decision, "Reject"),
# )

# mamdani = Mamdani(
#     rules=[rule_1, rule_2],
#     and_operator="min",
#     or_operator="max",
#     implication_operator="min",
#     aggregation_operator="max",
#     defuzzification_operator="cog",
# )


# plt.figure(figsize=(12, 8))
# mamdani.plot(score=185, ratio=0.25, credit=3)


#
# Tip Decision Problem
#

# from mf import *


# service_quality = FuzzyVariable(
#     name="service_quality",
#     universe=np.linspace(start=0, stop=10, num=200),
# )

# service_quality["poor"] = trimf(service_quality.universe, -1, 0, 5)
# service_quality["good"] = trimf(service_quality.universe, 0, 5, 10)
# service_quality["excellent"] = trimf(service_quality.universe, 5, 10, 15)

# food_quality = FuzzyVariable(
#     name="food_quality",
#     universe=np.linspace(start=0, stop=10, num=200),
# )

# food_quality["rancid"] = zmf(food_quality.universe, 0, 5)
# food_quality["delicious"] = smf(food_quality.universe, 5, 10)

# tip = FuzzyVariable(
#     name="tip",
#     universe=np.linspace(start=0, stop=25, num=200),
# )

# tip["small"] = trimf(tip.universe, -1, 0, 10)
# tip["average"] = trimf(tip.universe, 0, 10, 20)
# tip["generous"] = trapmf(tip.universe, 10, 20, 25, 30)


# rule_1 = FuzzyRule(
#     antecedents=[
#         (service_quality, "very", "poor"),
#         (food_quality, "extremely", "rancid"),
#     ],
#     consequent=(tip, "extremely", "small"),
#     is_and=False,
# )


# rule_2 = FuzzyRule(
#     antecedents=[
#         (service_quality, "good"),
#     ],
#     consequent=(tip, "average"),
# )


# rule_3 = FuzzyRule(
#     antecedents=[
#         (service_quality, "excellent"),
#         (food_quality, "delicious"),
#     ],
#     consequent=(tip, "generous"),
#     is_and=False,
# )

# mamdani = Mamdani(
#     rules=[
#         rule_1,
#         rule_2,
#         rule_3,
#     ],
#     and_operator="prod",
#     or_operator="max",
#     implication_operator="min",
#     aggregation_operator="max",
#     defuzzification_operator="cog",
# )

# plt.figure(figsize=(12, 8))
# mamdani.plot(
#     service_quality=2,
#     food_quality=2,
# )
