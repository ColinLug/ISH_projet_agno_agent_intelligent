"""Ce module permettra de préparer les données d'évaluation dont voici les instructions :

Préparation des données

Avant d'implémenter le framework, il est nécessaire de générer les données d'évaluation.
Concrètement, à partir des jeux de données déjà fournis, vous devez calculer les mesures de complexité hors ligne, c'est-à-dire avant de les fournir au modèle.

Pour savoir comment charger les données et calculer les mesures de complexité, référez-vous au fichier test.ipynb.
Notez que les fonctions implémentées constituent un exemple de base. La gestion des erreurs et des sauvegardes est laissée aux étudiant·e·s.

Vous devez également vérifier s'il existe des lignes qui satisfont déjà la condition de dominance, c'est-à-dire des cas où le texte simple est déjà plus complexe que le texte complexe.
Si de telles lignes existent, elles doivent être supprimées et leurs identifiants doivent être conservés.
Notez que chaque ligne doit garder son identifiant original, même si le jeu de données est modifié en supprimant certaines lignes.
"""
