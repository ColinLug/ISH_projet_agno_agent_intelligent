"""This module is the core of the project, it will help to create the workflow between the 2 agents"""

# Standard library imports
import json
import os
import sys
from collections import Counter
from functools import lru_cache
from pprint import pprint
from typing import Any, Dict, Iterable, Iterator, Optional, Set, Tuple

import nltk

# Third-party imports
import numpy as np
import pandas as pd
import spacy

# import stanza
import textcomplexity  # only used to access en.json
from agno.utils.pprint import pprint_run_response
from agno.workflow import Condition, Step, StepInput, StepOutput, Workflow
from dotenv import load_dotenv
from huggingface_hub.constants import HF_TOKEN_PATH
from nltk.corpus import wordnet as wn
from numpy.ma.core import true_divide
from tqdm.auto import tqdm

from check_superiority import ComplexityScore
from complexifier import complexifier_agent
from compute_superiority import (
    coherence_single,
    lexical_cohesion_single,
    lexical_measures_from_text,
    spacy_nlp,
    syntactic_measures_from_text,
)
from critic import critic_agent

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
datasets = {
    "ose_adv_ele": "data_sampled/OSE_adv_ele.txt",
    "ose_adv_int": "data_sampled/OSE_adv_int.txt",
    "swipe": "data_sampled/swipe.txt",
    "vikidia": "data_sampled/vikidia.txt",
}


def load_data(path):
    return pd.read_csv(path, sep="\t")


def load_dataset(name):
    if name not in datasets:
        raise ValueError(f"Dataset {name} not found")
    return load_data(datasets[name])


def needs_rewrite(step_input: StepInput) -> bool:
    text = step_input.input
    lex = lexical_measures_from_text(text)
    synt = syntactic_measures_from_text(text)
    new_comp_score = ComplexityScore(
        lex["MTLD"],
        lex["LD"],
        lex["LS"],
        synt["MDD"],
        synt["CS"],
        lexical_cohesion_single(text, spacy_nlp),
        coherence_single(text, spacy_nlp),
    )
    old_comp_score = ComplexityScore(0, 0, 0, 0, 0, 0, 0)
    return not new_comp_score > old_comp_score


complexification = Step(
    name="complexification",
    description="Complexify a sentence of a text",
    agent=complexifier_agent,
)
critic_complexification = Step(
    name="critic",
    description="Critic if needed the complexification of the sentence",
    agent=critic_agent,
)

complex_workflow = Workflow(
    name="Text Complexifier team",
    steps=[
        complexification,
        Condition(
            name="critic_condition",
            description="Check if a critic is needed",
            evaluator=needs_rewrite,
            steps=[critic_complexification],
        ),
    ],
)

pprint_run_response(
    complex_workflow.run(input="The cat sits on a mat", markdow=True, stream=True),
    markdown=True,
)
