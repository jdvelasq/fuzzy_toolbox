import numpy as np

from .variable import FuzzyVariable


class FuzzyRule:
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
                text += space + "AND " + antecedent[0].name + " IS"
                for k in range(1, len(antecedent)):
                    text += " " + antecedent[k]
                text += "\n"

        text += "THEN\n"
        text += space + self.get_consequent_name() + " IS"
        for k in range(1, len(self.consequent)):
            text += " " + self.consequent[k]
        return text

    def get_consequent_universe(self):
        return self.consequent[0].universe

    def get_consequent_membership(self):
        return self.consequent[0].sets[self.consequent[-1]]

    def get_consequent_name(self):
        return self.consequent[0].name

    def compute_inference(
        self, and_operator, or_operator, implication_operator, **values
    ):
        self.compute_memberships(**values)
        self.combine_inputs(and_operator, or_operator)
        self.compute_implication(implication_operator)

    def compute_memberships(self, **values):
        self.memberships = []

        for antecedent in self.antecedents:

            if len(antecedent) == 2:
                fuzzyvar, fuzzyset = antecedent
                modifier = None
                negation = False

            if len(antecedent) == 3:
                fuzzyvar, modifier, fuzzyset = antecedent
                if modifier.upper() == "NOT":
                    modifier = None
                    negation = True
                else:
                    negation = False

            if len(antecedent) == 4:
                fuzzyvar, negation, modifier, fuzzyset = antecedent
                negation = True

            crisp_value = values[fuzzyvar.name]
            membership = fuzzyvar.membership(crisp_value, fuzzyset, modifier, negation)
            self.memberships.append(membership)

    def combine_inputs(self, and_operator, or_operator):

        if len(self.memberships) > 1 and self.is_and is True:
            operator = {
                "min": np.min,
                "prod": np.prod,
            }[and_operator]
            self.combined_input = operator(self.memberships)
            return

        if len(self.memberships) > 1 and self.is_and is False:

            if or_operator == "min":
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

        self.combined_input = self.memberships

    def compute_implication(self, implication_operator):

        if len(self.consequent) == 2:
            modifier = None
            negation = False

        if len(self.consequent) == 3:
            modifier = self.consequent[1]
            if modifier.upper() == "NOT":
                modifier = None
                negation = True
            else:
                negation = False

        if len(self.consequent) == 4:
            modifier = self.consequent[2]
            negation = True

        fuzzyset = self.consequent[-1]
        membership = self.consequent[0].get_modified_membership(
            fuzzyset, modifier, negation
        )

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

# print(rule_1)
