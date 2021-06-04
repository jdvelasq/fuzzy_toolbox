from .rule import Rule
from .variable import FuzzyVariable


class Mamdani:
    """Mamdani inference system"""

    def __init__(
        self,
        rules,
        and_operator="min",
        implication_operator="min",
        aggregation_operator="max",
        defuzzification_operator="CoG",
    ):
        self.rules = rules
        self.and_operator = and_operator
        self.implication_operator = implication_operator
        self.aggregation_operator = aggregation_operator
        self.defuzzification_operator = defuzzification_operator
        #
        self.output = None
        self.values = None

    def __call__(self, **values):
        #
        # El proceso de calculo contiene siguientes pasos
        #
        self.values = values
        self.compute_rules()
        self.compute_aggregation()
        return self.compute_defuzzification()

    def compute_rules(self):
        #
        # Calcula el consecuente de cada regla usando el
        # operador de composición especificado.
        #
        for rule in self.rules:
            rule.compute_rule(
                and_operator=self.and_operator,
                implication_operator=self.implication_operator,
                **self.values
            )

    def compute_aggregation(self):

        #
        # Se geenera una variable difusa cuyos conjuntos
        # borrosos son los resultados de cada regla difusa
        #
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
        #
        # El conjunto borroso se convierte en un valor
        # crisp
        #
        return self.output.defuzzification(operator=self.defuzzification_operator)
