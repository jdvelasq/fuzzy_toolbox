import numpy as np

from .variable import FuzzyVariable


class FuzzyRule:
    def __init__(
        self,
        antecedents,
        consequent,
    ):
        self.antecedents = antecedents
        self.consequent = consequent
        self.memberships = None
        self.combined_input = None
        self.output = None

    def __repr__(self):
        text = "IF  "
        space = " " * 4
        for i, var in enumerate(self.antecedents):
            if i == 0:
                text += var[0].name + " IS " + var[1] + "\n"
            else:
                text += space + "AND " + var[0].name + " IS " + var[1] + "\n"

        text += "THEN\n"
        text += space + self.get_consequent_name() + " IS " + self.consequent[1]
        return text

    def get_consequent_universe(self):
        return self.consequent[0].universe

    def get_consequent_membership(self):
        return self.consequent[0].sets[self.consequent[1]]

    def get_consequent_name(self):
        return self.consequent[0].name

    def compute_inference(self, and_operator, implication_operator, **values):
        self.compute_memberships(**values)
        self.combine_inputs(and_operator)
        self.compute_implication(implication_operator)

    def compute_memberships(self, **values):
        self.memberships = []
        for fuzzyvar, fuzzyset in self.antecedents:
            if fuzzyvar.name in values.keys():
                crisp_value = values[fuzzyvar.name]
                membership = fuzzyvar.membership(crisp_value, fuzzyset)
                self.memberships.append(membership)

    def combine_inputs(self, and_operator):

        if len(self.memberships) > 1:
            operator = {
                "min": np.min,
                "prod": np.prod,
            }[and_operator]
            self.combined_input = operator(self.memberships)
        else:
            self.combined_input = self.memberships

    def compute_implication(self, implication_operator):

        membership = np.array(self.get_consequent_membership())

        if implication_operator == "min":
            membership = np.where(
                membership >= self.combined_input,
                self.combined_input,
                membership,
            )

        if implication_operator == "prod":
            membership = np.array(self.combined_input) * membership

        self.output = FuzzyVariable(
            name=self.get_consequent_name(),
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
