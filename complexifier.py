"""Ce script permettra de créer l'agent de complexification (cerveau bleu)"""

from agno.agent import Agent
from agno.models.ollama import Ollama

# Create each agent for each model
# Models name : Llama 3.1 8B / Qwen 2.5 7B / Mistral 7B / Falcon H1 7B
# Little models name : Gemma 3 / Qwen 4 / Mistral 7B / Llama 3.2
complexifier_agent = Agent(
    model=Ollama(,options={"temperature": 0.01})
)
"""Paramètres des modèles

Configurez les paramètres de génération suivants pour l'agent Text Complexification et pour l'agent Critique :

    Température : 0,01 (pour garantir une sortie presque déterministe).
    Top-p et top-k : utilisez les valeurs par défaut du modèle.
    Max tokens\max_length (longueur du texte généré) : définissez-le en fonction de la longueur de sortie pour chaque jeu de données. Pour chaque jeu de données, fixez cette valeur à 2 fois la longueur maximale des textes.
"""
