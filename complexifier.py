"""Ce script permettra de créer l'agent de complexification (cerveau bleu)"""

from agno.agent import Agent
from agno.models.huggingface import HuggingFace
from pydantic.main import BaseModel


# Création d'une nouvelle classe pour l'output en suivant un BaseModel (pas sûr que ce soit comme ça qu'on doit gérer les outputs)
class OutputJsonl(BaseModel):
    """"""

    def __init__(self):
        pass


# La longue description de l'agent donnée dans prompts.pdf
description_complex_agent = "[ROLE]\n You are a Text Complexification Assistant in a multi-agent framework for text complexification. You interact with a Critic Assistant that evaluates the complexity of your outputs and, when needed, sends you short, concrete action plans for revision in JSON format. Never flip roles. Never try to provide an action plan. Only the Critic Assistant is allowed to create or modify action plans. You and the Critic Assistant share a common interest in collaborating to successfully complete the task.\n Your task is to rewrite a given source text so that the generated result is more complex in lexicon, syntax, and discourse complexity, according to the guidelines provided in [COMPLEXITY GUIDELINES] and satisfying all the constraints in [OBJECTIVES].\nWhen you are given a source text in [SOURCE TEXT] with its current complexity profile in [TEXT COMPLEXITY PROFILE] and a target complexity profile in [TARGET COMPLEXITY PROFILE], you must generate a rewritten version whose complexity measures, as defined in [COMPLEXITY GUIDELINES], satisfy all the constraints defined in [OBJECTIVES].\n When you are given a previous version of your own output in [PREVIOUS TEXT] together with an [ACTION PLAN] produced by the Critic Assistant, you must apply only the specified actions in the plan to rewrite [PREVIOUS TEXT].\n You must strictly follow the task description, the objectives, the action plan (when given), and the output format specified in the current prompt. You must output only the rewritten text, as a continuous passage, with no explanations, no metric values, and no metacommentary. Never mention the multi-agent framework, the Critic Assistant, the guidelines, or the objectives in your output.\n [COMPLEXITY GUIDELINES]\n\n Lexical complexity (MTLD, LD, LS)\n [Definitions]\n\n Syntactic complexity (MDD, CS)\n[Definitions]\n\nDiscourse complexity (LC, CoH)\n[Definitions]\n\n[OUTPUT FORMAT]\nReturn only the rewritten text, with no additional headings, no metric values, and no meta text. Do not report any explanations."
# Je crois vraiment pas que ce sont des instructions prompts, à demander
instructions_complex_agent = [
    """[TASK]\nRewrite the text in [SOURCE TEXT], currently at complexity profile [TEXT COMPLEXITY PROFILE], so that the generated result demonstrates greater complexity in lexicon, syntax, and discourse structure, ultimately achieving and dominating the target complexity profile defined in [TARGET COMPLEXITY PROFILE] and satisfying all constraints in [OBJECTIVES]. Return the rewritten text as specified in [OUTPUT FORMAT].\n\n[TARGET COMPLEXITY PROFILE]\nThe target complexity profile you must dominate is [<>].\n\n[SOURCE TEXT]\n[<source text: paragraph or document>]\n\n[TEXT COMPLEXITY PROFILE]\nThe complexity profile of [SOURCE TEXT] is [<>], where the metrics are provided in the following fixed order: MTLD, LD, LS, MDD, CS, LC, CoH.\n\n[OBJECTIVES]\nThe rewritten text must achieve dominance over the target complexity profile, which is provided as an ordered vector of metrics: MTLD, LD, LS, MDD, CS, LC, CoH. Dominance is achieved in the following sense: every complexity measure (MTLD, LD, LS, MDD, CS, LC, CoH) of the generated text must be greater than or equal to its corresponding target value. Additionally, the generated text must provide strict improvement in at least one lexical dimension (MTLD, LD, or LS), at least one syntactic dimension (MDD or CS), and at least one discourse dimension (LC or CoH). The number of words of the generated text must be in the range [< here insert the 80% and the 120% of the number of words of the complex text >].""",
    """[TASK]\nRewrite your previously generated text in [PREVIOUS TEXT] by applying exactly and only the editing instructions contained in the ‘‘action plan’’ field of [ACTION PLAN]. Do not introduce any additional modifications beyond those specified in the plan. Return the rewritten text as specified in [OUTPUT FORMAT].\n\n[ACTION PLAN]\n[<JSON object returned by the Critic with "status": "revision required" and an ‘‘action plan’ array of 3--5 concrete actions specifying where and what to change>]\n\n[PREVIOUS TEXT]\n[<latest generated text by the Complexification Assistant>]""",
]


# Création des différents agents de complexification
# Les modèles lourds recommandés à tester en premier lieu : Llama 3.1 8B / Qwen 2.5 7B / Mistral 7B / Falcon H1 7B
complexifier_agent_llama = Agent(
    model=HuggingFace(
        id="meta-llama/Meta-Llama-3.1-8B-Instruct",  # meta-llama/llama-3.2-3b-instruct:free pour petit modèle
        name="Llama3.1",  # Llama3.2 pour petit modèle
        temperature=0.01,
        max_tokens=1000,  # une valeur par défaut
    ),
    description=description_complex_agent,
    instructions=instructions_complex_agent,
    output_schema=OutputJsonl,
)
complexifier_agent_qwen = Agent(
    model=HuggingFace(
        id="Qwen/Qwen2.5-7B-Instruct",  # qwen/qwen3-14b pour le petit modèle
        name="Qwen2.5",  # Qwen3 pour petit modèle
        temperature=0.01,
        max_tokens=1000,  # une valeur par défaut
    ),
    description=description_complex_agent,
    instructions=instructions_complex_agent,
    output_schema=OutputJsonl,
)
complexifier_agent_mistral = Agent(
    model=HuggingFace(
        id="mistralai/Mistral-7B-Instruct-v0.3",  # mistralai/mistral-7b-instruct:free pour petit modèle
        name="Mistral7B",
        temperature=0.01,
        max_tokens=1000,  # une valeur par défaut
    ),
    description=description_complex_agent,
    instructions=instructions_complex_agent,
    output_schema=OutputJsonl,
)
complexifier_agent_falcon = Agent(
    model=HuggingFace(
        id="tiiuae/Falcon-H1-7B-Instruct",  # google/gemma-3-12b-it:free pour petit modèle
        name="FalconH1",  # Gemma3
        temperature=0.01,
        max_tokens=1000,  # une valeur par défaut
    ),
    description=description_complex_agent,
    instructions=instructions_complex_agent,
    output_schema=OutputJsonl,
)
"""Paramètres des modèles

Configurez les paramètres de génération suivants pour l'agent Text Complexification et pour l'agent Critique :

    Température : 0,01 (pour garantir une sortie presque déterministe).
    Top-p et top-k : utilisez les valeurs par défaut du modèle.
    Max tokens\max_length (longueur du texte généré) : définissez-le en fonction de la longueur de sortie pour chaque jeu de données. Pour chaque jeu de données, fixez cette valeur à 2 fois la longueur maximale des textes.
"""
