class ComplexityScore:
    """Classe représentant un score de complexité
    """
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
        """Initialisation de l'objet ayant 3 attributs représentés par des dictionnaires
        Complexité lexicale, complexité syntaxique et complexité discursive

        Args:
            MTLD (float): le score MTLD
            LD (float): le score LD
            LS (float): le score LS
            MDD (float): le score MDD
            CS (float): le score CS
            LC (float): le score LC
            CoH (float): le score CoH
        """
        self.lex_comp = {"MTLD": MTLD, "LD": LD, "LS": LS}
        self.synt_comp = {"MDD": MDD, "CS": CS}
        self.disc_comp = {"LC": LC, "CoH": CoH}

    def __str__(self) -> str:
        """Permet d'afficher la classe correctement

        Returns:
            str: la chaîne de caractères formatée à afficher lors d'un print() p. ex.
        """
        return f"Lexical complexity : {self.lex_comp}\nSyntaxical complexity : {self.synt_comp}\nDiscursive complexity : {self.disc_comp}"

    def __eq__(self, other) -> bool:
        """Définit l'égalité entre deux objets de l'instance de la classe ComplexityScore

        Args:
            other (ComplexityScore): l'autre instance à laquelle comparer

        Returns:
            bool: la valeur de vérité de l'égalité
        """
        return (
            self.disc_comp == other.disc_comp
            and self.synt_comp == other.synt_comp
            and self.lex_comp == other.lex_comp
        )

    def __gt__(self, other) -> bool:
        """Définit la supériorité entre deux objets de l'instance de la classe ComplexityScore

        Args:
            other (ComplecityScore): l'autre instance à laquelle comparer

        Returns:
            bool: la valeur de vérité de la supériorité
        """
        # L'idée sera la même pour les 3 complexités (lexicale, syntaxique et discursive) :
        # On part de l'idée que _self_ a une complexité intermédiaire plus grande qu'_other_ et ce sera infirmé si:
        #   Il y a une égalité exacte dans la complexité intermédiaire
        #   Il y a un des scores de complexité de _self_ qui est plus petit que celui de _other_
        lex_result_bool = True
        if self.lex_comp == other.lex_comp:
            lex_result_bool = False
        else:
            for key in self.lex_comp:
                # Check si c'est bien un nombre (et pas un NaN)
                if type(self.lex_comp[key]) is type(other.lex_comp[key]) is float:
                    if self.lex_comp[key] < other.lex_comp[key]:
                        lex_result_bool = False
        
        # Voir commentaire plus haut
        synt_result_bool = True
        if self.synt_comp == other.synt_comp:
            synt_result_bool = False
        else:
            for key in self.synt_comp:
                # Check si c'est bien un nombre (et pas un NaN)
                if type(self.synt_comp[key]) is type(other.synt_comp[key]) is float:
                    if self.synt_comp[key] < other.synt_comp[key]:
                        synt_result_bool = False
        
        # Voir commentaire plus haut
        disc_result_bool = True
        if self.disc_comp == other.disc_comp:
            disc_result_bool = False
        else:
            for key in self.disc_comp:
                # Check si c'est bien un nombre (et pas un NaN)
                if type(self.disc_comp[key]) is type(other.disc_comp[key]) is float:
                    if self.disc_comp[key] < other.disc_comp[key]:
                        disc_result_bool = False

        # Fait un _et_ logique sur le résultat de chacune des complexités intermédiaires
        return lex_result_bool and synt_result_bool and disc_result_bool


# A supprimer dans le rendu final
a = ComplexityScore(0, 0, 0, 0, 0, 0, 0.1)
b = ComplexityScore(0.1, 0, 0, 0.3, 0, 0, 0.2)
print(a > b)
