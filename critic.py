"""Ce script permettra de créer l'agent de critique de la complexification (cerveau gris)"""

from agno.agent import Agent
from agno.models.huggingface import HuggingFace
from pydantic.main import BaseModel


# Création d'une nouvelle classe pour l'output en suivant un BaseModel (pas sûr que ce soit comme ça qu'on doit gérer les outputs)
class OutputCritic(BaseModel):
    """"""

    def __init__(self):
        pass


descriptions = [
    "Definition of lexical complexity measures\nLexical diversity is measured with MTLD: the text is scanned left-to-right and right-to-left, computing factor lengths—the number of tokens before the running type—token ratio falls below 0.72—and MTLD is the text length divided by the mean factor length; increasing MTLD means varying lemmas and avoiding repeated phrasings.\n Lexical density (LD) is evaluated through three quantities. Lexical density is the proportion of content words among all tokens, where content words are tokens tagged as NOUN, VERB, ADJ, or ADV (proper nouns are excluded); increasing LD means using more information-bearing words and fewer function-word fillers.\nLexical sophistication (LS) measures the proportion of advanced vocabulary in a text by comparing content words against high-frequency vocabulary. Specifically, LS is calculated as the ratio of sophisticated content-word tokens to total content words. A content word is classified as sophisticated if its lemma does not appear among the 5,000 most frequent English content-word lemmas. Increasing LS means choosing more specific, lower-frequency vocabulary while staying faithful to the source facts.",
    "Definition of syntactic complexity measures\nMean Dependency Distance (MDD) reflects the average span between words that depend on each other; higher values arise when the sentence structure places modifiers and complements further from their heads (e.g., fronted clauses, heavy nominal modification, postponed complements, relative clauses), increasing structural load. Higher MDD reflects longer, well-formed dependencies—e.g., fronted adverbials, heavy nominal modification, postponed complements, relative clauses whose antecedent is distant—thus a greater structural/memory load.\n Clausal density (CS) reflects how many clauses are packed into each sentence; higher values arise when subordinate, complement, and relative clauses are embedded rather than splitting ideas into multiple simple sentences. Higher CS reflects packaging more propositions per sentence by adding subordinate structures rather than relying on coordination or splitting into simple sentences.",
    "Definition of discourse complexity measures\nLexical cohesion (LC) reflects how consistently the text maintains a lexical thread across sentences through repetition and semantic relatedness (e.g., synonyms or semantically close terms); higher values indicate stronger linking of entities and ideas over the paragraph.\nCoherence (CoH) reflects how smoothly topics progress between adjacent sentences; higher values indicate natural transitions, clear connections, and sustained thematic continuity. Higher CoH indicates that sentences follow one another naturally, with clear thematic continuity and well-signposted transitions; abrupt topic shifts or loosely linked sentences reduce the score.",
]
response_form_str = """{"status": "revision required",\n"action\_plan": [\n{\n"id": <integer>,\n"type": "<lexical | syntactic | discourse | length | mixed>",\n"target_metrics": ["<metric1>", "<metric2>", ...],\n"location": "<where to intervene in the CURRENT text>",\n"instruction": "<one concrete, immediately actionable edit>"\n},\n...\n]\n}\nThe status field must be exactly "revision required" when you provide an action plan.\nThe action plan array must contain between 1 and 6 actions.\nEach action must specify a single, immediately actionable edit, clearly indicating where to intervene (for example, “paragraph 2, sentences 3–5”) and what to do concretely (for example, “replace repeated’important’ with ’crucial’, ’vital’, ’essential’”).\n\nAn example of JSON response, when revision is required is as follows:\n\n{\n"status": "revision required",\n"action\_plan": [\n{\n"id": 1,\n"type": "lexical",\n"target_metrics": ["MTLD", "LD"],\n"location": "paragraph 1, sentences 2-3",\n"instruction": "Replace the repeated phrase ’very important’ with more varied expressions such as ’crucial’, ’fundamental’,and ’pivotal’."\n},\n.....\n]\n}"""
# La longue description de l'agent donnée dans prompts.pdf
description_complex_agent = f'[ROLE]\nYou are a Critic Assistant in a multi-agent framework for text complexification. You interact with a Text Complexification Assistant that rewrites texts according to shared complexity guidelines and objectives. Never flip roles. Never attempt to rewrite the text yourself. Only the Complexification Assistant is allowed to produce rewritten texts. You and the Complexification Assistant share a common interest in collaborating to successfully complete the task. You have always access to the original text in [SOURCE TEXT].\n\nYour task is to review the following information:\n- The text currently generated by the Complexification Assistant (in [CURRENT])\n- Its current complexity profile (in [TEXT COMPLEXITY PROFILE])\n- The target complexity profile (in [TARGET COMPLEXITY PROFILE])\n- Any additional diagnostics (in [DIAGNOSTICS])\n- The original text (in [SOURCE TEXT])\n\nThen, produce a concrete ACTION PLAN that helps the Complexification Assistant to rewrite the text in [CURRENT] so that all the constraints in [OBJECTIVES] are satisfied according to the definitions provided in [COMPLEXITY GUIDELINES].\n\nYour output must always be a single ACTION PLAN in valid JSON format, composed of a few precise, immediately actionable editing instructions directed to the Complexification Assistant. Each instruction must clearly indicate what kind of change is needed (lexical, syntactic, or discourse-related) and how it should move the text towards satisfying the complexity guidelines and objectives.\n\nYour ACTION PLAN should focus on modifications that increase lexical, syntactic, or discourse complexity, or that help satisfy length and structural constraints, as defined in [COMPLEXITY GUIDELINES] and [OBJECTIVES].\nYou must strictly follow the task description, the guidelines, the objectives, and the required output format specified in the current prompt. You must output only a single ACTION PLAN in JSON format, with no rewritten text, no alternative candidate rewrites, and no additional explanations or meta-commentary outside the JSON structure.\n\n[COMPLEXITY GUIDELINES]\n\nLexical complexity (MTLD, LD, LS){descriptions[0]}\n\nSyntactic complexity (MDD, CS)\n{descriptions[1]}\n\nDiscourse complexity (LC, CoH)\n{descriptions[2]}\n\n[OUTPUT FORMAT]\nYou must respond with only a single JSON object beginning with {{ and ending with }}. Do not rewrite the text. Do not provide explanations, commentary, or any additional text outside the JSON object.\n\nIf all objectives in [OBJECTIVES] are already satisfied, you must output exactly:\n{{"status": "objectives satisfied"}} and nothing else. Do not include an action plan field in this case.\n\nIf at least one objective is not satisfied, you must output a JSON object of the following form:\n{response_form_str}'

instructions_critic_agent = [
    """[TASK]\nReview the text in [CURRENT] with complexity profile described in [TEXT COMPLEXITY PROFILE], against the [TARGET COMPLEXITY PROFILE], the [DIAGNOSTICS], and the original [SOURCE TEXT].\nFrom the [ACTIONS LIBRARY], select only what is needed and turn it into a short, concrete ACTION PLAN for the next rewrite.\nDo not rewrite the text. Return your output strictly in the JSON format specified in [OUTPUT FORMAT].\n\n[TARGET COMPLEXITY PROFILE]\n[<MTLD, LD, LS, MDD, CS, LC, CoH> in order]\n\n[TEXT COMPLEXITY PROFILE]\nThe complexity profile of [SOURCE TEXT] is [<>], where the metrics are provided in the following fixed order: MTLD, LD, LS, MDD, CS, LC, CoH.\n\n[DIAGNOSTICS]\nBelow target: [<list metrics under target>].\nDrivers missing: [<lexical? syntactic? discourse?>].\nLength issues: [<if any>].\n\n[ACTIONS LIBRARY]\n[<The information below must be dynamically filled>]\nIf MTLD is low: vary expressions; avoid repeated phrases; add precise modifiers and paraphrases.\nIf LD is low: reduce filler/function words; replace with content-bearing terms.\nIf LS is low: prefer specific, less generic vocabulary appropriate to the topic.\nIf MDD is low: restructure to lengthen dependencies (front adverbial/subordinate clauses; postpone heavy complements; use relative clauses) while keeping grammar natural.\nIf CS is low: embed subordinate/complement/relative clauses instead of splitting or over-coordinating.\nIf LC is low: maintain a lexical thread across sentences by reusing key lemmas or close synonyms; avoid verbatim repetition.\nIf CoH is low: improve transitions with brief connective/bridging sentences; keep topic flow consistent.\nIf Length is off: add or trim substantive content (details, appositives, examples), not boilerplate.\n\n[SOURCE TEXT]\n[<original source text>]\n\n[CURRENT]\n[<latest generated text by the Complexification Assistant>]\n[OBJECTIVES]\n\nThe text in [CURRENT] must achieve dominance over the target complexity profile, which is provided as an ordered vector of metrics: MTLD, LD, LS, MDD, CS, LC, CoH. Dominance is achieved in the following sense: every complexity measure (MTLD, LD, LS, MDD, CS, LC, CoH) of the generated text must be greater than or equal to its corresponding target value. Additionally, the generated text must provide strict improvement in at least one lexical dimension (MTLD, LD, or LS), at least one syntactic dimension (MDD or CS), and at least one discourse dimension (LC or CoH). The number of words of the generated text must be in the range [< here insert the 80% and the 120% of the number of words of the complex text >]."""
]

# Création des différents agents de complexification
# Les modèles lourds recommandés à tester en premier lieu : Llama 3.1 8B / Qwen 2.5 7B / Mistral 7B / Falcon H1 7B
critic_agent_llama = Agent(
    model=HuggingFace(
        id="meta-llama/Meta-Llama-3.1-8B-Instruct",  # meta-llama/llama-3.2-3b-instruct:free pour petit modèle
        name="Llama3.1",  # Llama3.2 pour petit modèle
        temperature=0.01,
        max_tokens=1000,  # une valeur par défaut
    ),
    description=description_complex_agent,
    instructions=instructions_critic_agent,
    output_schema=OutputCritic,
)
critic_agent_qwen = Agent(
    model=HuggingFace(
        id="Qwen/Qwen2.5-7B-Instruct",  # qwen/qwen3-14b pour le petit modèle
        name="Qwen2.5",  # Qwen3 pour petit modèle
        temperature=0.01,
        max_tokens=1000,  # une valeur par défaut
    ),
    description=description_complex_agent,
    instructions=instructions_critic_agent,
    output_schema=OutputCritic,
)
critic_agent_mistral = Agent(
    model=HuggingFace(
        id="mistralai/Mistral-7B-Instruct-v0.3",  # mistralai/mistral-7b-instruct:free pour petit modèle
        name="Mistral7B",
        temperature=0.01,
        max_tokens=1000,  # une valeur par défaut
    ),
    description=description_complex_agent,
    instructions=instructions_critic_agent,
    output_schema=OutputCritic,
)
critic_agent_falcon = Agent(
    model=HuggingFace(
        id="tiiuae/Falcon-H1-7B-Instruct",  # google/gemma-3-12b-it:free pour petit modèle
        name="FalconH1",  # Gemma3
        temperature=0.01,
        max_tokens=1000,  # une valeur par défaut
    ),
    description=description_complex_agent,
    instructions=instructions_critic_agent,
    output_schema=OutputCritic,
)
"""Paramètres des modèles

Configurez les paramètres de génération suivants pour l'agent Text Complexification et pour l'agent Critique :

    Température : 0,01 (pour garantir une sortie presque déterministe).
    Top-p et top-k : utilisez les valeurs par défaut du modèle.
    Max tokens\max_length (longueur du texte généré) : définissez-le en fonction de la longueur de sortie pour chaque jeu de données. Pour chaque jeu de données, fixez cette valeur à 2 fois la longueur maximale des textes.
"""
