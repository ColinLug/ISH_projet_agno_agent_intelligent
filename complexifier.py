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
instructions_complex_agent = [
    "Definition of lexical complexity measures\nLexical diversity is measured with MTLD: the text is scanned left-to-right and right-to-left, computing factor lengths—the number of tokens before the running type—token ratio falls below 0.72—and MTLD is the text length divided by the mean factor length; increasing MTLD means varying lemmas and avoiding repeated phrasings.\n Lexical density (LD) is evaluated through three quantities. Lexical density is the proportion of content words among all tokens, where content words are tokens tagged as NOUN, VERB, ADJ, or ADV (proper nouns are excluded); increasing LD means using more information-bearing words and fewer function-word fillers.\nLexical sophistication (LS) measures the proportion of advanced vocabulary in a text by comparing content words against high-frequency vocabulary. Specifically, LS is calculated as the ratio of sophisticated content-word tokens to total content words. A content word is classified as sophisticated if its lemma does not appear among the 5,000 most frequent English content-word lemmas. Increasing LS means choosing more specific, lower-frequency vocabulary while staying faithful to the source facts.",
    "Definition of syntactic complexity measures\nMean Dependency Distance (MDD) reflects the average span between words that depend on each other; higher values arise when the sentence structure places modifiers and complements further from their heads (e.g., fronted clauses, heavy nominal modification, postponed complements, relative clauses), increasing structural load. Higher MDD reflects longer, well-formed dependencies—e.g., fronted adverbials, heavy nominal modification, postponed complements, relative clauses whose antecedent is distant—thus a greater structural/memory load.\n Clausal density (CS) reflects how many clauses are packed into each sentence; higher values arise when subordinate, complement, and relative clauses are embedded rather than splitting ideas into multiple simple sentences. Higher CS reflects packaging more propositions per sentence by adding subordinate structures rather than relying on coordination or splitting into simple sentences.",
    "Definition of discourse complexity measures\nLexical cohesion (LC) reflects how consistently the text maintains a lexical thread across sentences through repetition and semantic relatedness (e.g., synonyms or semantically close terms); higher values indicate stronger linking of entities and ideas over the paragraph.\nCoherence (CoH) reflects how smoothly topics progress between adjacent sentences; higher values indicate natural transitions, clear connections, and sustained thematic continuity. Higher CoH indicates that sentences follow one another naturally, with clear thematic continuity and well-signposted transitions; abrupt topic shifts or loosely linked sentences reduce the score.",
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
