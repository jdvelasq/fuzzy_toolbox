"""
Fuzzy Rules
==============================================================================

"""
import matplotlib.pyplot as plt
import numpy as np

from .variable import FuzzyVariable


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
        antecedents,
        consequent,
        is_and=True,
    ):
        self.antecedents = antecedents
        self.consequent = consequent
        self.is_and = is_and
        self.memberships = None
        self.combined_input = None
        self.output = None
        self.memberships = None

    def __repr__(self):
        text = "IF  "
        space = " " * 4
        for i, antecedent in enumerate(self.antecedents):

            if i == 0:
                text += antecedent[0].name + " IS"
                for k in range(1, len(antecedent)):
                    text += " " + antecedent[k]
                text += "\n"
            else:
                if self.is_and is True:
                    text += space + "AND " + antecedent[0].name + " IS"
                else:
                    text += space + "OR " + antecedent[0].name + " IS"
                for k in range(1, len(antecedent)):
                    text += " " + antecedent[k]
                text += "\n"

        text += "THEN\n"
        text += space + self.get_consequent_name() + " IS"
        for k in range(1, len(self.consequent)):
            text += " " + self.consequent[k]
        return text

    def get_consequent_universe(self):
        """Gets the universe of the fuzzy variable in the consquent."""

        return self.consequent[0].universe

    def get_consequent_membership(self):
        """Gets the membership of the fuzzy variable in the consquent."""
        return self.consequent[0].sets[self.consequent[-1]]

    def get_consequent_name(self):
        """Gets the name of the fuzzy variable in the consquent."""
        return self.consequent[0].name

    def compute_inference(
        self, and_operator, or_operator, implication_operator, **values
    ):
        """Computes the output fuzzy set of the rule.

        Args:
            and_operator (string): {"min"|"prod"}
            or_operator (string): {"max"|"probor"}
            implication_operator (string): {"min"|"prod"}
            values (dict): Crisp values for the antecedent variables.

        """
        self.compute_memberships(**values)
        self.combine_inputs(and_operator, or_operator)
        self.compute_implication(implication_operator)

    def compute_memberships(self, **values):
        """Computes the memberships of the antecedents.

        Args:
            values: crisp values for the antecedentes in the rule.
        """
        self.memberships = []

        for antecedent in self.antecedents:

            if len(antecedent) == 2:
                fuzzyvar, fuzzyset = antecedent
                modifiers = None
            else:
                fuzzyvar = antecedent[0]
                fuzzyset = antecedent[-1]
                modifiers = antecedent[1:-1]

            crisp_value = values[fuzzyvar.name]
            membership = fuzzyvar.compute_membership(crisp_value, fuzzyset, modifiers)
            self.memberships.append(membership)

    def combine_inputs(self, and_operator, or_operator):
        """Computes the firing strength of the rule.

        Args:
            and_operator (string): {"min"|"prod"}
            or_operator (string): {"max"|"probor"}

        """

        if len(self.memberships) > 1 and self.is_and is True:
            operator = {
                "min": np.min,
                "prod": np.prod,
            }[and_operator]
            self.combined_input = operator(self.memberships)
            return

        if len(self.memberships) > 1 and self.is_and is False:

            if or_operator == "max":
                self.combined_input = np.min(self.memberships)
                return

            if or_operator == "probor":
                self.combined_input = self.memberships[0]
                for i in range(1, len(self.memberships)):
                    self.combined_input = (
                        self.combined_input
                        + self.memberships[i]
                        - self.combined_input * self.memberships[i]
                    )
                return

            self.combined_input = None

        self.combined_input = self.memberships

    def compute_implication(self, implication_operator):
        """Computes the implication of the rule.

        Args:
            implication_operator (string): {"min"|"prod"}

        """

        if len(self.consequent) == 2:
            modifiers = None
        else:
            modifiers = self.consequent[1:-1]

        fuzzyset = self.consequent[-1]
        membership = self.consequent[0].get_modified_membership(fuzzyset, modifiers)
        self.output_membership = membership

        if implication_operator == "min":
            membership = np.where(
                membership >= self.combined_input,
                self.combined_input,
                membership,
            )

        if implication_operator == "prod":
            membership = np.array(self.combined_input) * membership

        membership = np.array(membership)

        self.output = FuzzyVariable(
            name="Implication",
            universe=self.get_consequent_universe(),
            sets={"rule_output": membership},
        )

    def plot(
        self,
        and_operator="prod",
        or_operator="max",
        implication_operator="prod",
        n_antecedents=None,
        n_rows=1,
        i_row=0,
        **values
    ):
        def format_plot(title):
            plt.gca().set_ylim(-0.05, 1.05)
            plt.gca().spines["left"].set_visible(False)
            plt.gca().spines["bottom"].set_visible(True)
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)
            plt.gca().get_yaxis().set_visible(False)
            plt.gca().get_xaxis().set_visible(True)

            if title is not None:
                plt.gca().set_title(title)

        if n_antecedents is None:
            n_antecedents = len(self.antecedents)

        self.compute_inference(
            and_operator=and_operator,
            or_operator=or_operator,
            implication_operator=implication_operator,
            **values,
        )

        universes = {
            antecedent[0].name: antecedent[0].universe
            for antecedent in self.antecedents
        }
        self.compute_memberships(**universes)

        for i, antecedent in enumerate(self.antecedents):

            value = values[antecedent[0].name]

            plt.subplot(n_rows, n_antecedents + 1, i_row * (n_antecedents + 1) + i + 1)
            plt.gca().plot(
                antecedent[0].universe, self.memberships[i], "-k", linewidth=1
            )
            membership = np.interp(
                x=values[antecedent[0].name],
                xp=universes[antecedent[0].name],
                fp=self.memberships[i],
            )
            membership = np.where(
                self.memberships[i] <= membership, self.memberships[i], membership
            )
            plt.gca().fill_between(
                antecedent[0].universe, membership, color="gray", alpha=0.7
            )

            format_plot(
                title="{} = {}".format(antecedent[0].name, value),
            )

            plt.gca().vlines(x=value, ymin=-0.0, ymax=1.0, color="red", linewidth=2)
            if i == 0:
                plt.gca().get_yaxis().set_visible(True)

        plt.subplot(
            n_rows, n_antecedents + 1, i_row * (n_antecedents + 1) + n_antecedents + 1
        )
        plt.gca().plot(self.output.universe, self.output_membership, "-", color="gray")

        plt.gca().plot(
            self.output.universe, self.output["rule_output"], "-k", linewidth=2
        )
        plt.gca().fill_between(
            self.output.universe,
            self.output["rule_output"],
            color="gray",
            alpha=0.7,
        )

        format_plot(title=self.consequent[0].name)

        plt.gca().get_yaxis().set_visible(True)
        plt.gca().yaxis.tick_right()


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
#     antecedents=[
#         (score, "High"),
#         (ratio, "Goodr"),
#         (credit, "Goodc"),
#     ],
#     consequent=(decision, "Approve"),
# )

# plt.figure(figsize=(12, 3))
# rule_1.plot(score=180, ratio=0.25, credit=3)

# # print(rule_1)
