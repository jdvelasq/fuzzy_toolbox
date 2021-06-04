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
