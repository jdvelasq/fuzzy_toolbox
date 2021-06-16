"""
Zade-Mamdani fuzzy model
===============================================================================

"""
import matplotlib.pyplot as plt
import numpy as np

from .core import (
    apply_modifiers,
    plot_crisp_input,
    plot_fuzzy_input,
    plot_fuzzyvariable,
)

# #############################################################################
#
#
# Fuzzy Variable
#
#
# #############################################################################


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

    def plot(self, fmt="-", linewidth=2):
        """Plots the fuzzy sets defined for the variable.

        Args:
            figsize (tuple): figure size.

        """
        plot_fuzzyvariable(
            universe=self.universe,
            memberships=[self.sets[k] for k in self.sets.keys()],
            labels=list(self.sets.keys()),
            title=self.name,
            fmt=fmt,
            linewidth=linewidth,
            view_xaxis=True,
            view_yaxis=True,
        )

    def plot_input(self, value, fuzzyset, view_xaxis=True, view_yaxis="left"):

        if isinstance(value, (np.ndarray, list)):

            plot_fuzzy_input(
                value=value,
                universe=self.universe,
                membership=self.sets[fuzzyset],
                name=self.name,
                view_xaxis=view_xaxis,
                view_yaxis=view_yaxis,
            )

        else:

            plot_crisp_input(
                value=value,
                universe=self.universe,
                membership=self.sets[fuzzyset],
                name=self.name,
                view_xaxis=view_xaxis,
                view_yaxis=view_yaxis,
            )

    def apply_modifiers(self, fuzzyset, modifiers):
        """Computes a modified membership function.

        Args:
            fuzzyset (string): Identifier of the fuzzy set.
            modifiers (list of string): {"very"|"somewhat"|"more_or_less"|"extremely"|"plus"|"intensify"|"slightly"|None}

        Returns:
            A numpy.array.

        """
        if modifiers is None:
            return self.sets[fuzzyset]

        return apply_modifiers(membership=self.sets[fuzzyset], modifiers=modifiers)

    def fuzzificate(self, value, fuzzyset, modifiers=None):
        """Computes the value of the membership function on a specifyied point of the universe for the fuzzy set.

        Args:
            value (float, numpy.array): point to evaluate the value of the membership function.
            fuzzyset (string): name of the fuzzy set.
            modifier (string): membership function modifier.
            negation (bool): returns the negation?.

        Returns:
            A float number or numpy.array.
        """

        membership = self.apply_modifiers(fuzzyset, modifiers)

        return np.interp(
            x=value,
            xp=self.universe,
            fp=membership,
        )


# #############################################################################
#
#
# Operators over fuzzy sets
#
#
# #############################################################################


def defuzzificate(universe, membership, operator="cog"):
    """Computes a representative crisp value for the fuzzy set.

    Args:
        fuzzyset (string): Fuzzy set to defuzzify
        operator (string): {"cog"|"bisection"|"mom"|"lom"|"som"}

    Returns:
        A float value.

    """

    def cog():
        start = np.min(universe)
        stop = np.max(universe)
        x = np.linspace(start, stop, num=200)
        m = np.interp(x, xp=universe, fp=membership)
        return np.sum(x * m) / sum(m)

    def coa():
        start = np.min(universe)
        stop = np.max(universe)
        x = np.linspace(start, stop, num=200)
        m = np.interp(x, xp=universe, fp=membership)
        area = np.sum(m)
        cum_area = np.cumsum(m)
        return np.interp(area / 2, xp=cum_area, fp=x)

    def mom():
        maximum = np.max(membership)
        maximum = np.array([u for u, m in zip(universe, membership) if m == maximum])
        return np.mean(maximum)

    def lom():
        maximum = np.max(membership)
        maximum = np.array([u for u, m in zip(universe, membership) if m == maximum])
        return np.max(maximum)

    def som():
        maximum = np.max(membership)
        maximum = np.array([u for u, m in zip(universe, membership) if m == maximum])
        return np.min(maximum)

    if np.sum(membership) == 0.0:
        return 0.0

    return {
        "cog": cog,
        "coa": coa,
        "mom": mom,
        "lom": lom,
        "som": som,
    }[operator]()


def aggregate(memberships, operator):
    """Replace the fuzzy sets by a unique fuzzy set computed by the aggregation operator.

    Args:
        operator (string): {"max"|"sim"|"probor"} aggregation operator.

    Returns:
        A FuzzyVariable

    """
    result = memberships[0]

    if operator == "max":
        for membership in memberships[1:]:
            result = np.maximum(result, membership)
        return result

    if operator == "sum":
        for membership in memberships[1:]:
            result = result + membership
        return np.minimum(1, result)

    if operator == "probor":
        for membership in memberships[1:]:
            result = result + membership - result * membership
        return np.maximum(1, np.minimum(1, result))


# #############################################################################
#
#
# Zadeh-Mamdani's Rule
#
#
# #############################################################################


class FuzzyRule:
    """Mamdani fuzzy rule.

    Creates a Mamdani fuzzy fule.

    Args:
        antecedents (list of tuples): Fuzzy variables in the rule antecedent.
        consequent (tuple): Fuzzy variable in the consequence.
        is_and (bool): When True, membership values are combined using the specified AND operator; when False, the OR operator is used.

    """

    def __init__(
        self,
        premises,
        consequence,
    ):
        self.premises = premises
        self.consequence = consequence
        #

    def __repr__(self):

        text = "IF  "
        space = " " * 4

        for i_premise, premise in enumerate(self.premises):

            if i_premise == 0:
                text += premise[0].name + " IS"
                for t in premise[1:]:
                    text += " " + t
                text += "\n"
            else:
                text += space + premise[0] + " " + premise[1].name + " IS"
                for t in premise[2:]:
                    text += " " + t
                text += "\n"

        text += "THEN\n"
        text += space + self.consequence[0].name + " IS"
        for t in self.consequence[1:]:
            text += " " + t
        return text


# #############################################################################
#
#
# Generic inference method
#
#
# #############################################################################


def probor(a, b):
    return np.maximum(0, np.minimum(1, a + b - a * b))


class InferenceMethod:
    def __init__(
        self,
        and_operator,
        or_operator,
        composition_operator,
        production_link,
        defuzzification_operator,
    ):
        self.and_operator = and_operator
        self.or_operator = or_operator
        self.composition_operator = composition_operator
        self.production_link = production_link
        self.defuzzification_operator = defuzzification_operator
        #
        self.rules = []

    def compute_modified_premise_memberships(self):

        for rule in self.rules:

            rule.modified_premise_memberships = {}
            rule.universes = {}

            for i_premise, premise in enumerate(rule.premises):

                if i_premise == 0:

                    if len(premise) == 2:
                        fuzzyvar, fuzzyset = premise
                        modifiers = None
                    else:
                        fuzzyvar = premise[0]
                        fuzzyset = premise[-1]
                        modifiers = premise[1:-1]
                else:

                    if len(premise) == 3:
                        _, fuzzyvar, fuzzyset = premise
                        modifiers = None
                    else:
                        fuzzyvar = premise[1]
                        fuzzyset = premise[-1]
                        modifiers = premise[2:-1]

                rule.modified_premise_memberships[
                    fuzzyvar.name
                ] = fuzzyvar.apply_modifiers(fuzzyset, modifiers)

                rule.universes[fuzzyvar.name] = fuzzyvar.universe

    def compute_modified_consequence_membership(self):

        for rule in self.rules:

            if len(rule.consequence) == 2:
                modifiers = None
            else:
                modifiers = rule.consequence[1:-1]

            fuzzyset = rule.consequence[-1]

            rule.modified_consequence_membership = rule.consequence[0].apply_modifiers(
                fuzzyset, modifiers
            )

    def build_infered_consequence(self):

        self.infered_consequence = FuzzyVariable(
            name=self.rules[0].consequence[0].name,
            universe=self.rules[0].consequence[0].universe,
        )

        for i_rule, rule in enumerate(self.rules):

            self.infered_consequence["Rule-{}".format(i_rule)] = rule.infered_membership

    def aggregate_productions(self):
        """Computes the output fuzzy set of the inference system."""

        infered_membership = None

        if self.production_link == "max":

            for rule in self.rules:
                if infered_membership is None:
                    infered_membership = rule.infered_membership
                else:
                    infered_membership = np.maximum(
                        infered_membership, rule.infered_membership
                    )

        self.infered_membership = infered_membership

    def fuzzificate(self, **values):
        """Computes the memberships of the antecedents.

        Args:
            values: crisp values for the antecedentes in the rule.
        """

        for rule in self.rules:

            rule.fuzzificated_values = {}
            for name in rule.modified_premise_memberships.keys():
                crisp_value = values[name]
                rule.fuzzificated_values[name] = np.interp(
                    x=crisp_value,
                    xp=rule.universes[name],
                    fp=rule.modified_premise_memberships[name],
                )

    def defuzzificate(self):

        self.defuzzificated_infered_membership = defuzzificate(
            universe=self.infered_consequence.universe,
            membership=self.infered_membership,
            operator=self.defuzzification_operator,
        )


# #############################################################################
#
#
# Decompositional Inference Method
#
#
# #############################################################################


class DecompositionalInference(InferenceMethod):
    def __init__(
        self,
        input_type,
        and_operator,
        or_operator,
        implication_operator,
        composition_operator,
        production_link,
        defuzzification_operator,
    ):
        super().__init__(
            and_operator=and_operator,
            or_operator=or_operator,
            composition_operator=composition_operator,
            production_link=production_link,
            defuzzification_operator=defuzzification_operator,
        )

        self.implication_operator = implication_operator
        self.input_type = input_type
        #

    def __call__(self, rules, **values):

        self.rules = rules

        self.compute_modified_premise_memberships()
        self.compute_modified_consequence_membership()
        self.compute_fuzzy_implication()

        if self.input_type == "crisp":
            self.fuzzificate(**values)
        else:
            self.fuzzificated_values = values.copy()

        self.compute_fuzzy_composition()
        self.compute_aggregation()
        self.build_infered_consequence()
        self.aggregate_productions()
        self.defuzzificate()

        return self.defuzzificated_infered_membership

    def compute_fuzzy_implication(self):

        #
        # Implication operators
        # See Kasabov, pag. 185
        #
        Ra = lambda u, v: np.minimum(1, 1 - u + v)
        Rm = lambda u, v: np.maximum(np.minimum(u, v), 1 - u)
        Rc = lambda u, v: np.minimum(u, v)
        Rb = lambda u, v: np.maximum(1 - u, v)
        Rs = lambda u, v: np.where(u <= v, 1, 0)
        Rg = lambda u, v: np.where(u <= v, 1, v)
        Rsg = lambda u, v: np.minimum(Rs(u, v), Rg(1 - u, 1 - v))
        Rgs = lambda u, v: np.minimum(Rg(u, v), Rs(1 - u, 1 - v))
        Rgg = lambda u, v: np.minimum(Rg(u, v), Rg(1 - u, 1 - v))
        Rss = lambda u, v: np.minimum(Rs(u, v), Rs(1 - u, 1 - v))

        implication_fn = {
            "Ra": Ra,
            "Rm": Rm,
            "Rc": Rc,
            "Rb": Rb,
            "Rs": Rs,
            "Rg": Rg,
            "Rsg": Rsg,
            "Rgs": Rgs,
            "Rgg": Rgg,
            "Rss": Rss,
        }[self.implication_operator]

        for rule in self.rules:

            rule.fuzzy_implications = {}

            for name in rule.modified_premise_memberships.keys():

                premise_membership = rule.modified_premise_memberships[name]
                consequence_membership = rule.modified_consequence_membership
                V, U = np.meshgrid(consequence_membership, premise_membership)
                rule.fuzzy_implications[name] = implication_fn(U, V)

    def compute_fuzzy_composition(self):

        for rule in self.rules:

            rule.fuzzy_compositions = {}

            for name in rule.modified_premise_memberships.keys():

                implication = rule.fuzzy_implications[name]

                if self.input_type == "fuzzy":
                    #
                    # Fuzzy imput
                    #
                    value = self.fuzzificated_values[name]
                    n_dim = len(value)
                    value = value.reshape((n_dim, 1))
                    value = np.tile(value, (1, implication.shape[1]))
                else:
                    #
                    # Crisp input
                    #
                    value = rule.fuzzificated_values[name]
                    value = value * np.ones(shape=implication.shape)

                if self.composition_operator == "min":
                    composition = np.minimum(value, implication)
                if self.composition_operator == "prod":
                    composition = value * implication

                rule.fuzzy_compositions[name] = composition.max(axis=0)

    def compute_aggregation(self):

        for rule in self.rules:

            aggregated_membership = None

            for premise in rule.premises:

                if aggregated_membership is None:
                    aggregated_membership = rule.fuzzy_compositions[premise[0].name]
                else:
                    other_membership = rule.fuzzy_compositions[premise[1].name]

                    if premise[0] == "AND":
                        if self.and_operator == "min":
                            aggregated_membership = np.minimum(
                                aggregated_membership, other_membership
                            )
                        if self.and_operator == "prod":
                            aggregated_membership = (
                                aggregated_membership * other_membership
                            )

                    if premise[0] == "OR":
                        if self.and_operator == "max":
                            aggregated_membership = np.maximum(
                                aggregated_membership, other_membership
                            )
                        if self.and_operator == "probor":
                            aggregated_membership = probor(
                                aggregated_membership, other_membership
                            )

            rule.infered_membership = aggregated_membership

    def plot(self, rules, **values):
        def get_position():
            names = []
            for rule in rules:
                for i_premise, premise in enumerate(rule.premises):
                    if i_premise == 0:
                        names.append(premise[0].name)
                    else:
                        names.append(premise[1].name)
            names = sorted(set(names))
            position = {name: i_name for i_name, name in enumerate(names)}
            return position

        # computation
        self.__call__(rules, **values)

        n_rows = len(self.rules) + 1
        position = get_position()
        n_variables = len(position.keys())

        for i_rule, rule in enumerate(rules):

            #
            # Plot premises
            #
            for i_premise, premise in enumerate(rule.premises):

                if i_premise == 0:
                    varname = premise[0].name
                else:
                    varname = premise[1].name

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

                if self.input_type == "crisp":
                    plot_crisp_input(
                        value=values[varname],
                        universe=rule.universes[varname],
                        membership=rule.modified_premise_memberships[varname],
                        name=title,
                        view_xaxis=view_xaxis,
                        view_yaxis=view_yaxis,
                    )
                else:
                    plot_fuzzy_input(
                        value=values[varname],
                        universe=rule.universes[varname],
                        membership=rule.modified_premise_memberships[varname],
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

            plot_fuzzy_input(
                value=rule.infered_membership,
                universe=rule.consequence[0].universe,
                membership=rule.modified_consequence_membership,
                name=None,  # rule.consequence[0].name,
                view_xaxis=False,
                view_yaxis="right",
            )

        plt.subplot(
            n_rows,
            n_variables + 1,
            n_rows * (n_variables + 1),
        )

        plot_crisp_input(
            value=self.defuzzificated_infered_membership,
            universe=self.infered_consequence.universe,
            membership=self.infered_membership,
            name=None,
            view_xaxis=True,
            view_yaxis="right",
        )
        plt.gca().set_xlabel(
            "{} = {:.2f}".format(
                self.infered_consequence.name, self.defuzzificated_infered_membership
            )
        )


# #############################################################################
#
#
# Fuzz-Infer-Defuzz Method
#
#
# #############################################################################


class FIDInference(InferenceMethod):
    def __init__(
        self,
        and_operator,
        or_operator,
        composition_operator,
        production_link,
        defuzzification_operator,
    ):
        super().__init__(
            and_operator=and_operator,
            or_operator=or_operator,
            composition_operator=composition_operator,
            production_link=production_link,
            defuzzification_operator=defuzzification_operator,
        )

    def __call__(self, rules, **values):

        self.rules = rules
        self.compute_modified_premise_memberships()
        self.compute_modified_consequence_membership()
        self.fuzzificate(**values)
        self.aggregate_premises()
        self.compute_composition()
        self.build_infered_consequence()
        self.aggregate_productions()
        self.defuzzificate()

        return self.defuzzificated_infered_membership

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

    def compute_composition(self):
        """Computes the rule composition for all rules."""

        for rule in self.rules:

            if len(rule.consequence) == 2:
                modifiers = None
            else:
                modifiers = rule.consequence[1:-1]

            fuzzyset = rule.consequence[-1]
            infered_membership = rule.consequence[0].apply_modifiers(
                fuzzyset, modifiers
            )

            if self.composition_operator == "min":
                infered_membership = np.where(
                    infered_membership >= rule.aggregated_membership,
                    rule.aggregated_membership,
                    infered_membership,
                )

            if self.composition_operator == "prod":
                infered_membership = (
                    np.array(rule.aggregated_membership) * infered_membership
                )

            rule.infered_membership = infered_membership

    def plot(self, rules, **values):
        def get_position():
            names = []
            for rule in rules:
                for i_premise, premise in enumerate(rule.premises):
                    if i_premise == 0:
                        names.append(premise[0].name)
                    else:
                        names.append(premise[1].name)
            names = sorted(set(names))
            position = {name: i_name for i_name, name in enumerate(names)}
            return position

        # computation
        self.__call__(rules, **values)

        n_rows = len(self.rules) + 1
        position = get_position()
        n_variables = len(position.keys())

        for i_rule, rule in enumerate(rules):

            #
            # Plot premises
            #
            for i_premise, premise in enumerate(rule.premises):

                if i_premise == 0:
                    varname = premise[0].name
                else:
                    varname = premise[1].name

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

                plot_crisp_input(
                    value=values[varname],
                    universe=rule.universes[varname],
                    membership=rule.modified_premise_memberships[varname],
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

            plot_fuzzy_input(
                value=rule.infered_membership,
                universe=rule.consequence[0].universe,
                membership=rule.modified_consequence_membership,
                name=None,  # rule.consequence[0].name,
                view_xaxis=True,
                view_yaxis="right",
            )

        plt.subplot(
            n_rows,
            n_variables + 1,
            n_rows * (n_variables + 1),
        )

        plot_crisp_input(
            value=self.defuzzificated_infered_membership,
            universe=self.infered_consequence.universe,
            membership=self.infered_membership,
            name=None,
            view_xaxis=True,
            view_yaxis="right",
        )
        plt.gca().set_xlabel(
            "{} = {:.2f}".format(
                self.infered_consequence.name, self.defuzzificated_infered_membership
            )
        )


# from mf import trimf

# score = FuzzyVariable(
#     name="score",
#     universe=np.arange(start=150, stop=201, step=5),
#     sets={
#         "High": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.7, 1.0, 1.0, 1.0],
#         "Low": [1.0, 1.0, 0.8, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
#     premises=[
#         (score, "High"),
#         ("AND", ratio, "Goodr"),
#         ("AND", credit, "Goodc"),
#     ],
#     consequence=(decision, "Approve"),
# )


# print(rule_1)


# rule_2 = FuzzyRule(
#     premises=[
#         (score, "Low"),
#         ("AND", ratio, "Badr"),
#         ("OR", credit, "Badc"),
#     ],
#     consequence=(decision, "Reject"),
# )

# model = FIDInference(
#     and_operator="min",
#     or_operator="max",
#     composition_operator="min",
#     production_link="max",
#     defuzzification_operator="cog",
# )


# model = DecompositionalInference(
#     input_type="crisp",
#     and_operator="min",
#     or_operator="max",
#     implication_operator="Rc",
#     composition_operator="min",  # min / prod
#     production_link="max",
#     defuzzification_operator="cog",
# )


# print(
#     model(
#         rules=[rule_1, rule_2],
#         score=190,
#         ratio=0.39,
#         credit=1.5,
#     )
# )

# plt.figure(figsize=(12, 9))
# model.plot(
#     rules=[rule_1, rule_2],
#     score=190,
#     ratio=0.39,
#     credit=1.5,
# )


# model = DecompositionalInference(
#     input_type="fuzzy",
#     and_operator="min",
#     or_operator="max",
#     implication_operator="Rc",
#     composition_operator="min",  # min / prod
#     production_link="max",
#     defuzzification_operator="cog",
# )


# score_1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 1.0])
# ratio_1 = np.array([1, 1, 0.6, 0.2, 0, 0, 0, 0, 0, 0, 0])
# credit_1 = np.array([0, 1, 1, 0.7, 0.3, 0, 0, 0, 0, 0, 0])

# score_2 = np.array([0.9, 0.7, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# ratio_2 = np.array([0, 0, 0, 0, 0, 0, 0, 0.3, 0.5, 0.7, 0.9])
# credit_2 = np.array([0, 0, 0, 0, 0, 0, 0, 0.3, 0.5, 0.7, 0.9])

# score_3 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.6, 0.8])
# ratio_3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0.4, 0.6, 0.8])
# credit_3 = np.array([1, 1, 1, 0.8, 0.6, 0.4, 0, 0, 0, 0, 0])


# plt.figure(figsize=(12, 9))
# model.plot(
#     rules=[rule_1, rule_2],
#     score=score_3,
#     ratio=ratio_3,
#     credit=credit_3,
# )


# service_quality = Variable(
#     name="service_quality",
#     universe=np.linspace(start=0, stop=10, num=20),
# )

# service_quality["poor"] = trimf(service_quality.universe, -1, 0, 5)
# service_quality["good"] = trimf(service_quality.universe, 0, 5, 10)
# service_quality["excellent"] = trimf(service_quality.universe, 5, 10, 15)

# # service_quality.plot()
# # service_quality.plot_input(
# #     value=3, fuzzyset="poor", view_yaxis="right", view_xaxis=False
# # )

# x = [max(0, min(1, 3 - 0.2 * i)) for i in range(20)]
# print(x)
# service_quality.plot_input(
#     value=x, fuzzyset="poor", view_yaxis="right", view_xaxis=False
# )


#
#
#
#
#


# #
# # Test: Credit Decision Problem
# #
# # score = FuzzyVariable(
# #     name="score",
# #     universe=np.arange(start=150, stop=201, step=5),
# #     sets={
# #         "High": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.7, 1.0, 1.0, 1.0],
# #         "Low": [1.0, 1.0, 0.8, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
# #     },
# # )

# # ratio = FuzzyVariable(
# #     name="ratio",
# #     universe=[0.1, 0.3, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.5, 0.7, 1.0],
# #     sets={
# #         "Goodr": [1, 1, 0.7, 0.3, 0, 0, 0, 0, 0, 0, 0],
# #         "Badr": [0, 0, 0, 0, 0, 0, 0, 0.3, 0.7, 1.0, 1.0],
# #     },
# # )

# # credit = FuzzyVariable(
# #     name="credit",
# #     universe=list(range(11)),
# #     sets={
# #         "Goodc": [1, 1, 1, 0.7, 0.3, 0, 0, 0, 0, 0, 0],
# #         "Badc": [0, 0, 0, 0, 0, 0, 0.3, 0.7, 1, 1, 1],
# #     },
# # )

# # decision = FuzzyVariable(
# #     name="decision",
# #     universe=list(range(11)),
# #     sets={
# #         "Approve": [0, 0, 0, 0, 0, 0, 0.3, 0.7, 1, 1, 1],
# #         "Reject": [1, 1, 1, 0.7, 0.3, 0, 0, 0, 0, 0, 0],
# #     },
# # )

# # rule_1 = Mamdani(
# #     antecedents=[
# #         (score, "High"),
# #         (ratio, "Goodr"),
# #         (credit, "Goodc"),
# #     ],
# #     consequent=(decision, "Approve"),
# # )

# # rule_2 = Mamdani(
# #     antecedents=[
# #         (score, "Low"),
# #         (ratio, "Badr"),
# #         (credit, "Badc"),
# #     ],
# #     consequent=(decision, "Reject"),
# # )

# # mamdani = Mamdani(
# #     rules=[rule_1, rule_2],
# #     and_operator="min",
# #     or_operator="max",
# #     implication_operator="min",
# #     aggregation_operator="max",
# #     defuzzification_operator="cog",
# # )


# # plt.figure(figsize=(12, 8))
# # mamdani.plot(score=185, ratio=0.25, credit=3)


# #
# # Tip Decision Problem
# #

# # from mf import *


# # service_quality = FuzzyVariable(
# #     name="service_quality",
# #     universe=np.linspace(start=0, stop=10, num=200),
# # )

# # service_quality["poor"] = trimf(service_quality.universe, -1, 0, 5)
# # service_quality["good"] = trimf(service_quality.universe, 0, 5, 10)
# # service_quality["excellent"] = trimf(service_quality.universe, 5, 10, 15)

# # food_quality = FuzzyVariable(
# #     name="food_quality",
# #     universe=np.linspace(start=0, stop=10, num=200),
# # )

# # food_quality["rancid"] = zmf(food_quality.universe, 0, 5)
# # food_quality["delicious"] = smf(food_quality.universe, 5, 10)

# # tip = FuzzyVariable(
# #     name="tip",
# #     universe=np.linspace(start=0, stop=25, num=200),
# # )

# # tip["small"] = trimf(tip.universe, -1, 0, 10)
# # tip["average"] = trimf(tip.universe, 0, 10, 20)
# # tip["generous"] = trapmf(tip.universe, 10, 20, 25, 30)


# # rule_1 = Mamdani(
# #     antecedents=[
# #         (service_quality, "very", "poor"),
# #         (food_quality, "extremely", "rancid"),
# #     ],
# #     consequent=(tip, "extremely", "small"),
# #     is_and=False,
# # )


# # rule_2 = Mamdani(
# #     antecedents=[
# #         (service_quality, "good"),
# #     ],
# #     consequent=(tip, "average"),
# # )


# # rule_3 = Mamdani(
# #     antecedents=[
# #         (service_quality, "excellent"),
# #         (food_quality, "delicious"),
# #     ],
# #     consequent=(tip, "generous"),
# #     is_and=False,
# # )

# # mamdani = Mamdani(
# #     rules=[
# #         rule_1,
# #         rule_2,
# #         rule_3,
# #     ],
# #     and_operator="prod",
# #     or_operator="max",
# #     implication_operator="min",
# #     aggregation_operator="max",
# #     defuzzification_operator="cog",
# # )

# # plt.figure(figsize=(12, 8))
# # mamdani.plot(
# #     service_quality=2,
# #     food_quality=2,
# # )
