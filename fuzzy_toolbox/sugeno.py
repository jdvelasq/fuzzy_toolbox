"""
Sugeno fuzzy model
==============================================================================

"""
import matplotlib.pyplot as plt
import numpy as np


from .core import plot_crisp_input


# #############################################################################
#
#
# Fuzzy Variable
#
#
# #############################################################################


class FuzzyVariable:
    def __init__(self, name, universe, sets):
        self.name = name.replace(" ", "_")
        self.universe = universe
        self.sets = sets

    def compute_membership(self, value, fuzzyset):
        fn, params = self.sets[fuzzyset]
        membership = fn(value, **params)
        return membership


class SugenoRule:
    def __init__(self, premises, coef, intercept):
        self.premises = premises
        self.coef = coef
        self.intercept = intercept

    def __repr__(self):

        text = "IF\n"
        space = " " * 4

        for i_premise, premise in enumerate(self.premises):

            if i_premise == 0:
                text += space + premise[0].name + " IS"
                for t in premise[1:]:
                    text += " " + t
                text += "\n"
            else:
                text += space + premise[0] + " " + premise[1].name + " IS"
                for t in premise[2:]:
                    text += " " + t
                text += "\n"

        text += "THEN\n"
        if self.coef is not None:
            for i_key, key in enumerate(self.coef.keys()):

                if i_key == 0:
                    text += space
                    if self.coef[key] > 0:
                        if self.coef[key] == 1:
                            text += key + "\n"
                        else:
                            text += str(self.coef[key]) + " " + key + "\n"
                    if self.coef[key] < 0:
                        if self.coef[key] == -1:
                            text += "- " + key + "\n"
                        else:
                            text += "- " + str(abs(self.coef[key])) + " " + key + "\n"
                else:
                    if self.coef[key] > 0:
                        if self.coef[key] == 1:
                            text += space + "+ " + key + "\n"
                        else:
                            text += (
                                space + "+ " + str(self.coef[key]) + " " + key + "\n"
                            )
                    if self.coef[key] < 0:
                        if self.coef[key] == -1:
                            text += space + "- " + key + "\n"
                        else:
                            text += (
                                space
                                + "- "
                                + str(abs(self.coef[key]))
                                + " "
                                + key
                                + "\n"
                            )

            if self.intercept is not None:
                if self.intercept > 0:
                    text += space + "+ " + str(self.intercept)
                if self.intercept < 0:
                    text += space + "- " + str(abs(self.intercept))

        else:

            if self.intercept is not None:
                if self.intercept >= 0:
                    text += space + str(self.intercept)
                if self.intercept < 0:
                    text += space + "- " + str(abs(self.intercept))

        return text


def probor(a, b):
    return np.maximum(0, np.minimum(1, a + b - a * b))


class Sugeno:
    def __init__(self, and_operator, or_operator, defuzzification_operator):
        self.and_operator = and_operator
        self.or_operator = or_operator
        self.defuzzification_operator = defuzzification_operator
        #
        self.implication_operator = "prod"

    def __call__(self, rules, **values):

        self.rules = rules
        # self.get_universes()
        self.fuzzificate(**values)
        self.aggregate_premises()
        self.build_infered_consequence(**values)
        self.aggregate_productions()

        return self.infered_value

    def fuzzificate(self, **values):
        """Computes the memberships of the antecedents.

        Args:
            values: crisp values for the antecedentes in the rule.
        """

        for rule in self.rules:

            rule.fuzzificated_values = {}

            for i_premise, premise in enumerate(rule.premises):

                if i_premise == 0:
                    fuzzyvar, fuzzyset = premise
                else:
                    _, fuzzyvar, fuzzyset = premise

                crisp_value = values[fuzzyvar.name]
                fn, params = fuzzyvar.sets[fuzzyset]
                rule.fuzzificated_values[fuzzyvar.name] = fn(
                    crisp_value, **params
                ).tolist()

    def aggregate_premises(self):

        for rule in self.rules:

            if isinstance(self.or_operator, str):
                or_operator = {
                    "max": np.maximum,
                    "probor": probor,
                }[self.or_operator]

            if isinstance(self.and_operator, str):
                and_operator = {
                    "min": np.minimum,
                    "prod": np.multiply,
                }[self.and_operator]

            rule.aggregated_membership = None

            for premise in rule.premises:

                if rule.aggregated_membership is None:
                    rule.aggregated_membership = rule.fuzzificated_values[
                        premise[0].name
                    ]
                else:
                    name = premise[1].name
                    if premise[0] == "OR":
                        rule.aggregated_membership = or_operator(
                            rule.aggregated_membership, rule.fuzzificated_values[name]
                        )
                    if premise[0] == "AND":
                        rule.aggregated_membership = and_operator(
                            rule.aggregated_membership, rule.fuzzificated_values[name]
                        )

    def build_infered_consequence(self, **values):

        for rule in self.rules:

            result = 0

            for key in rule.coef.keys():
                crisp_value = values[key]
                coef = rule.coef[key]
                result += coef * crisp_value

            if rule.intercept is not None:
                result += result.intercept

            rule.infered_consequence = result

    def aggregate_productions(self):

        if self.defuzzification_operator == "wtaver":
            s = sum([rule.aggregated_membership for rule in self.rules])
            for rule in self.rules:
                rule.aggregated_membership = rule.aggregated_membership / s

        infered_value = 0
        for rule in self.rules:
            infered_value += rule.aggregated_membership * rule.infered_consequence

        self.infered_value = infered_value

    def plot(self, rules, **values):
        #
        def get_position():
            names = []
            for rule in rules:
                for i_premise, premise in enumerate(rule.premises):
                    if i_premise == 0:
                        names.append(premise[0].name)
                    else:
                        names.append(premise[1].name)

                if rule.coef is not None:
                    for key in rule.coef.keys():
                        names.append(key)

            names = sorted(set(names))
            position = {name: i_name for i_name, name in enumerate(names)}
            return position

        # computation
        self.__call__(rules, **values)

        n_rows = len(self.rules) + 1
        position = get_position()
        n_variables = len(position.keys())

        universes = {}

        for i_rule, rule in enumerate(rules):

            #
            # Plot premises
            #
            for i_premise, premise in enumerate(rule.premises):

                if i_premise == 0:
                    fuzzyvar = premise[0]
                    varname = premise[0].name
                    fuzzyset = premise[1]
                else:
                    fuzzyvar = premise[1]
                    varname = premise[1].name
                    fuzzyset = premise[2]

                i_col = position[varname]

                if i_col == 0:
                    view_yaxis = "left"
                else:
                    view_yaxis = False

                plt.subplot(
                    n_rows,
                    n_variables + 1,
                    i_rule * (n_variables + 1) + i_col + 1,
                )

                view_xaxis = True if i_rule + 1 == len(rules) else False
                title = varname if i_rule == 0 else None

                #
                fn, params = fuzzyvar.sets[fuzzyset]
                universe = fuzzyvar.universe
                membership = fn(universe, **params)
                #
                universes[varname] = universe
                #

                plot_crisp_input(
                    value=values[varname],
                    universe=universe,
                    membership=membership,
                    name=title,
                    view_xaxis=view_xaxis,
                    view_yaxis=view_yaxis,
                )

            #
            # Plot consesquence
            #
            plt.subplot(
                n_rows,
                n_variables + 1,
                i_rule * (n_variables + 1) + n_variables + 1,
            )

            text_kwargs = dict(ha="center", va="bottom", fontsize=18, color="black")
            plt.gca().text(
                0.5, 0.5, str(round(rule.infered_consequence, 2)), **text_kwargs
            )
            text_kwargs = dict(ha="center", va="top", fontsize=18, color="black")
            plt.gca().text(
                0.5,
                0.5,
                "(" + str(round(rule.aggregated_membership, 2)) + ")",
                **text_kwargs
            )

            plt.gca().get_yaxis().set_visible(False)
            plt.gca().get_xaxis().set_visible(False)

        plt.subplot(
            n_rows,
            n_variables + 1,
            n_rows * (n_variables + 1),
        )

        text_kwargs = dict(ha="center", va="center", fontsize=18, color="black")
        plt.gca().text(0.5, 0.5, str(round(self.infered_value, 2)), **text_kwargs)
        plt.gca().get_yaxis().set_visible(False)
        plt.gca().get_xaxis().set_visible(False)

        #
        # format first column
        #
        for i_row in range(len(self.rules)):
            plt.subplot(
                n_rows,
                n_variables + 1,
                i_row * (n_variables + 1) + 1,
            )

            plt.gca().set_ylim(-0.05, 1.05)
            plt.gca().spines["bottom"].set_visible(True)
            plt.gca().spines["left"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)
            plt.gca().spines["top"].set_visible(False)

        for i_col in range(len(position.keys())):
            plt.subplot(
                n_rows,
                n_variables + 1,
                (n_rows - 2) * (n_variables + 1) + i_col + 1,
            )

            varname = [k for k in position.keys() if position[k] == i_col][0]
            xmin = min(universes[varname])
            xmax = max(universes[varname])

            plt.gca().set_xlim(xmin, xmax)
            plt.gca().spines["bottom"].set_visible(True)
            plt.gca().spines["left"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)
            plt.gca().spines["top"].set_visible(False)


# from mf import trimf

# x1 = FuzzyVariable(
#     name="x1",
#     universe=np.linspace(start=0, stop=20, num=50),
#     sets={
#         "small_1": (trimf, {"a": 0, "b": 0, "c": 16}),
#         "big_1": (trimf, {"a": 10, "b": 20, "c": 20}),
#     },
# )

# # print(x1.compute_membership(0, "small_1"))

# x2 = FuzzyVariable(
#     name="x2",
#     universe=np.linspace(start=0, stop=20, num=50),
#     sets={
#         "small_2": (trimf, {"a": 0, "b": 0, "c": 8}),
#         "big_2": (trimf, {"a": 2, "b": 10, "c": 20}),
#     },
# )

# rule_1 = SugenoRule(
#     premises=[
#         (x1, "small_1"),
#         ("AND", x2, "small_2"),
#     ],
#     coef={"x1": 1, "x2": 1},
#     intercept=None,
# )

# rule_2 = SugenoRule(
#     premises=[
#         (x1, "big_1"),
#     ],
#     coef={"x1": 2},
#     intercept=None,
# )

# rule_3 = SugenoRule(
#     premises=[
#         (x2, "big_2"),
#     ],
#     coef={"x2": 3},
#     intercept=None,
# )

# # print(rule_1)
# # print()
# # print(rule_2)
# # print()
# # print(rule_3)


# model = Sugeno(
#     and_operator="min",
#     or_operator="max",
#     defuzzification_operator="wtaver",
# )

# plt.figure(figsize=(12, 9))
# model.plot(
#     rules=[rule_1, rule_2, rule_3],
#     x1=12,
#     x2=5,
# )
