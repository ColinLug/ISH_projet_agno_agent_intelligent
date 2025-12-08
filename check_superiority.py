class ComplexityScore:
    def __init__(
        self,
        MTLD: float,
        LD: float,
        LS: float,
        MDD: float,
        CS: float,
        LC: float,
        CoH: float,
    ) -> None:
        self.lex_comp = {"MTLD": MTLD, "LD": LD, "LS": LS}
        self.synt_comp = {"MDD": MDD, "CS": CS}
        self.disc_comp = {"LC": LC, "CoH": CoH}

    def __str__(self) -> str:
        return f"Lexical complexity : {self.lex_comp}\nSyntaxical complexity : {self.synt_comp}\nDiscursive complexity : {self.disc_comp}"

    def __eq__(self, other) -> bool:
        return (
            self.disc_comp == other.disc_comp
            and self.synt_comp == other.synt_comp
            and self.lex_comp == other.lex_comp
        )

    def __gt__(self, other) -> bool:
        lex_result_bool = True
        if self.lex_comp == other.lex_comp:
            lex_result_bool = False
        else:
            for key in self.lex_comp:
                # Check si c'est bien un nombre (et pas un NaN)
                if type(self.lex_comp[key]) is type(other.lex_comp[key]) is float:
                    if self.lex_comp[key] < other.lex_comp[key]:
                        lex_result_bool = False

        synt_result_bool = True
        if self.synt_comp == other.synt_comp:
            synt_result_bool = False
        else:
            for key in self.synt_comp:
                # Check si c'est bien un nombre (et pas un NaN)
                if type(self.synt_comp[key]) is type(other.synt_comp[key]) is float:
                    if self.synt_comp[key] < other.synt_comp[key]:
                        synt_result_bool = False

        disc_result_bool = True
        if self.disc_comp == other.disc_comp:
            disc_result_bool = False
        else:
            for key in self.disc_comp:
                # Check si c'est bien un nombre (et pas un NaN)
                if type(self.disc_comp[key]) is type(other.disc_comp[key]) is float:
                    if self.disc_comp[key] < other.disc_comp[key]:
                        disc_result_bool = False

        return lex_result_bool and synt_result_bool and disc_result_bool


# A supprimer dans le rendu final
a = ComplexityScore(0, 0, 0, 0, 0, 0, 0.1)
b = ComplexityScore(0.1, 0, 0, 0.3, 0, 0, 0.2)
print(a > b)
