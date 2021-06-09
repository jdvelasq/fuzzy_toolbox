"""
Mamdani fuzzy model
==============================================================================

"""

from .rule import FuzzyRule
from .variable import FuzzyVariable


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
                **self.values
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
        self.output.aggregate()

    def compute_defuzzification(self):
        """Computes the equivalent crisp value representing the output fuzzy set of the system."""
        return self.output.defuzzification(operator=self.defuzzification_operator)
