# Standard library imports
import importlib.resources as pkg_resources
import json
import sys
from collections import Counter
from functools import lru_cache
from pprint import pprint
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import nltk

# Third-party imports
import numpy as np
import pandas as pd
import spacy
import stanza
import textcomplexity  # only used to access en.json
from nltk.corpus import wordnet as wn
from tqdm.auto import tqdm

# Download required resources
stanza.download("en")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Make sure WordNet is available; if not, download it.
try:
    _ = wn.synsets("dog")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")
# Load spaCy model
nlp = spacy.load("en_core_web_md")
spacy_nlp = nlp
spacy_nlp.add_pipe("sentencizer")
# Cache stanza pipelines to avoid re-loading models
_STANZA_PIPELINES: Dict[str, stanza.Pipeline] = {}

# UPOS tags considered content words (C)
CONTENT_UPOS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}


@lru_cache()
def load_cow_top5000_en() -> Set[str]:
    """
    Load the COW-based list of the 5,000 most frequent English content words
    from textcomplexity's English language definition file (en.json).

    We ignore POS tags and keep only lowercased word forms.
    """
    with (
        pkg_resources.files(textcomplexity)
        .joinpath("en.json")
        .open("r", encoding="utf-8") as f
    ):
        lang_def = json.load(f)

    most_common = lang_def["most_common"]  # list of [word, xpos]
    cow_top5000 = {w.lower() for w, xpos in most_common}
    return cow_top5000


def get_stanza_pipeline(lang: str = "en", use_gpu: bool = False) -> stanza.Pipeline:
    """
    Get (or create) a cached stanza Pipeline for a given language.

    NOTE: You must have downloaded the models beforehand, e.g.:
        import stanza
        stanza.download('en')
    """
    if lang not in _STANZA_PIPELINES:
        _STANZA_PIPELINES[lang] = stanza.Pipeline(
            lang=lang,
            processors="tokenize,pos,lemma,depparse,constituency",
            use_gpu=use_gpu,
            tokenize_no_ssplit=False,
        )
    return _STANZA_PIPELINES[lang]


def _compute_mtld(
    tokens: Iterable[str], ttr_threshold: float = 0.72
) -> Optional[float]:
    """
    Compute MTLD (Measure of Textual Lexical Diversity) for a list of tokens.

    MTLD = total_number_of_tokens / number_of_factors

    A factor is a contiguous segment where the running TTR stays >= threshold.
    When the TTR drops below the threshold, we close a factor (at the previous
    token) and start a new one. At the end, the remaining partial segment is
    counted as a fractional factor, with weight proportional to how close the
    final TTR is to the threshold.
    """
    tokens = [tok for tok in tokens if tok]
    if not tokens:
        return None

    types = set()
    factor_count = 0.0
    token_count_in_factor = 0

    for tok in tokens:
        token_count_in_factor += 1
        types.add(tok)
        ttr = len(types) / token_count_in_factor

        if ttr < ttr_threshold:
            factor_count += 1.0
            types = set()
            token_count_in_factor = 0

    # final partial factor
    if token_count_in_factor > 0:
        final_ttr = len(types) / token_count_in_factor
        if final_ttr < 1.0:
            fractional = (1.0 - final_ttr) / (1.0 - ttr_threshold)
            fractional = max(0.0, min(1.0, fractional))
            factor_count += fractional

    if factor_count == 0:
        return None

    return len(tokens) / factor_count


def _compute_lexical_density(total_tokens: int, content_tokens: int) -> Optional[float]:
    """
    LD = |C| / |T|
    where:
        |C| = number of content-word tokens
        |T| = total number of non-punctuation tokens
    """
    if total_tokens == 0:
        return None
    return content_tokens / total_tokens


def _compute_lexical_sophistication_cow(
    content_forms: Iterable[str],
    cow_top5000: set,
) -> Optional[float]:
    """
    LS = |{ w in C : w not in R }| / |C|
    where:
        C = content-word tokens (surface forms, lowercased)
        R = COW top-5000 content word forms (lowercased)
    """
    forms = [f for f in content_forms if f]
    if not forms:
        return None

    off_list = sum(1 for f in forms if f not in cow_top5000)
    return off_list / len(forms)


def lexical_measures_from_doc(doc) -> Dict[str, Optional[float]]:
    """
    Compute MTLD, LD, LS from a stanza Document.
    """
    cow_top5000 = load_cow_top5000_en()

    mtld_tokens = []
    total_tokens = 0
    content_tokens = 0
    content_forms = []

    for sent in doc.sentences:
        for word in sent.words:
            if word.upos == "PUNCT":
                continue

            lemma = (word.lemma or word.text or "").lower()
            if not lemma:
                continue

            mtld_tokens.append(lemma)
            total_tokens += 1

            if word.upos in CONTENT_UPOS:
                content_tokens += 1
                form = (word.text or "").lower()
                content_forms.append(form)

    mtld = _compute_mtld(mtld_tokens) if mtld_tokens else None
    ld = _compute_lexical_density(total_tokens, content_tokens)
    ls = _compute_lexical_sophistication_cow(content_forms, cow_top5000)

    return {"MTLD": mtld, "LD": ld, "LS": ls}


def lexical_measures_from_text(
    text: str, lang: str = "en"
) -> Dict[str, Optional[float]]:
    """
    Convenience wrapper: parse a single text and compute lexical measures.
    """
    if text is None:
        text = ""
    text = str(text)

    if not text.strip():
        return {"MTLD": None, "LD": None, "LS": None}

    nlp = get_stanza_pipeline(lang)
    doc = nlp(text)
    return lexical_measures_from_doc(doc)


def compute_lexical_measures_df(
    df: pd.DataFrame,
    column: str = "text",
    lang: str = "en",
) -> Dict[str, Dict[Any, Optional[float]]]:
    """
    Compute lexical measures for each row in df[column].

    Returns:
        {
            "MTLD": {index: value},
            "LD":   {index: value},
            "LS":   {index: value},
        }
    """
    mtld_res: Dict[Any, Optional[float]] = {}
    ld_res: Dict[Any, Optional[float]] = {}
    ls_res: Dict[Any, Optional[float]] = {}

    for idx, text in df[column].items():
        metrics = lexical_measures_from_text(text, lang=lang)
        mtld_res[idx] = metrics["MTLD"]
        ld_res[idx] = metrics["LD"]
        ls_res[idx] = metrics["LS"]

    return {"MTLD": mtld_res, "LD": ld_res, "LS": ls_res}


def mdd_from_doc(doc) -> Optional[float]:
    """
    Compute Mean Dependency Distance (MDD) from a stanza Document.

    For each sentence s_i with dependency set D_i:
        MDD_i = (1 / |D_i|) * sum_{(h,d) in D_i} |h - d|
    Then:
        MDD = (1 / k) * sum_i MDD_i, over all sentences with at least one dependency.
    """
    sentence_mdds = []

    for sent in doc.sentences:
        distances = []
        for w in sent.words:
            if w.head is None or w.head == 0:
                continue
            distances.append(abs(w.id - w.head))

        if distances:
            sentence_mdds.append(sum(distances) / len(distances))

    if not sentence_mdds:
        return None
    return sum(sentence_mdds) / len(sentence_mdds)


def _count_clauses_in_tree(tree) -> int:
    """
    Count clause nodes in a constituency tree.

    A simple and standard heuristic (PTB-style) is:
        count all nodes whose label starts with 'S'
        (S, SBAR, SBARQ, SINV, SQ, etc.).

    This aligns with the idea of counting finite and subordinate clauses
    as in Hunt (1965) and later complexity work.
    """
    if tree is None:
        return 0

    # Stanza's constituency tree: tree.label, tree.children
    count = 1 if getattr(tree, "label", "").startswith("S") else 0

    for child in getattr(tree, "children", []):
        # leaves can be strings or terminals without 'label'
        if hasattr(child, "label"):
            count += _count_clauses_in_tree(child)

    return count


def cs_from_doc(doc) -> Optional[float]:
    """
    Compute CS (clauses per sentence) from a stanza Document.

        CS = (1 / k) * sum_i L_i

    where L_i is the number of clauses in sentence s_i, estimated by counting
    all constituents whose label starts with 'S' in the constituency tree of s_i.
    """
    clause_counts = []
    for sent in doc.sentences:
        tree = getattr(sent, "constituency", None)
        if tree is None:
            # No constituency tree available for this sentence
            continue
        num_clauses = _count_clauses_in_tree(tree)
        clause_counts.append(num_clauses)

    if not clause_counts:
        return None

    return sum(clause_counts) / len(clause_counts)


def syntactic_measures_from_doc(doc) -> Dict[str, Optional[float]]:
    """
    Compute MDD and CS from a stanza Document.
    """
    mdd = mdd_from_doc(doc)
    cs = cs_from_doc(doc)
    return {"MDD": mdd, "CS": cs}


def syntactic_measures_from_text(
    text: str, lang: str = "en"
) -> Dict[str, Optional[float]]:
    """
    Convenience wrapper: parse a single text and compute syntactic measures.
    """
    if text is None:
        text = ""
    text = str(text)

    if not text.strip():
        return {"MDD": None, "CS": None}

    nlp = get_stanza_pipeline(lang)
    doc = nlp(text)
    return syntactic_measures_from_doc(doc)


def compute_syntactic_measures_df(
    df: pd.DataFrame,
    column: str = "text",
    lang: str = "en",
) -> Dict[str, Dict[Any, Optional[float]]]:
    """
    Compute syntactic measures for each row in df[column].

    Returns:
        {
            "MDD": {index: value},
            "CS":  {index: value},
        }
    """
    mdd_res: Dict[Any, Optional[float]] = {}
    cs_res: Dict[Any, Optional[float]] = {}

    for idx, text in df[column].items():
        metrics = syntactic_measures_from_text(text, lang=lang)
        mdd_res[idx] = metrics["MDD"]
        cs_res[idx] = metrics["CS"]

    return {"MDD": mdd_res, "CS": cs_res}


# Approximate set of content POS tags (spaCy universal POS)
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}


def is_content_token(tok):
    """
    Return True if token is considered a content word.
    We ignore stopwords, punctuation, and non-alphabetic tokens.
    """
    return tok.is_alpha and not tok.is_stop and tok.pos_ in CONTENT_POS


@lru_cache(maxsize=100000)
def get_related_lemmas(lemma):
    """
    Return a set of semantically related lemmas for the given lemma
    using WordNet, including:
      - synonyms
      - antonyms
      - hypernyms / hyponyms
      - meronyms (part/member/substance)
      - coordinate terms (siblings under the same hypernym)

    NOTE: Some older examples mention 'troponyms', but in NLTK's
    WordNet interface there is no 'troponyms()' method on Synset,
    so we do NOT use it here.
    """
    lemma = lemma.lower()
    related = set()
    synsets = wn.synsets(lemma)

    for syn in synsets:
        # Synonyms and antonyms
        for l in syn.lemmas():
            related.add(l.name().lower().replace("_", " "))
            for ant in l.antonyms():
                related.add(ant.name().lower().replace("_", " "))

        # Hypernyms (more general) and hyponyms (more specific)
        for hyper in syn.hypernyms():
            for l in hyper.lemmas():
                related.add(l.name().lower().replace("_", " "))
        for hypo in syn.hyponyms():
            for l in hypo.lemmas():
                related.add(l.name().lower().replace("_", " "))

        # Meronyms: part/member/substance
        for mer in (
            syn.part_meronyms() + syn.member_meronyms() + syn.substance_meronyms()
        ):
            for l in mer.lemmas():
                related.add(l.name().lower().replace("_", " "))

        # Coordinate terms (siblings under same hypernym)
        for hyper in syn.hypernyms():
            for sibling in hyper.hyponyms():
                if sibling == syn:
                    continue
                for l in sibling.lemmas():
                    related.add(l.name().lower().replace("_", " "))

    # Remove the lemma itself if present
    related.discard(lemma)
    return related


def lexical_cohesion_single(text, nlp):
    """
    Compute Lexical Cohesion (LC) for a single document:

        LC = |C| / m

    where:
      - |C| is the number of cohesive devices between sentences
        (lexical repetition + semantic relations),
      - m  is the total number of word tokens (alphabetic) in the document.

    If the document has fewer than 2 sentences or no valid words,
    LC is returned as 0.0.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    doc = nlp(text)

    # Total number of alphabetic tokens (denominator m)
    m = sum(1 for tok in doc if tok.is_alpha)
    if m == 0:
        return 0.0

    sentences = list(doc.sents)
    if len(sentences) < 2:
        # With only one sentence, cross-sentence cohesion is not defined
        return 0.0

    # Collect sets of content lemmas per sentence
    sent_lemmas = []
    for sent in sentences:
        lemmas = set(tok.lemma_.lower() for tok in sent if is_content_token(tok))
        if lemmas:
            sent_lemmas.append(lemmas)

    if len(sent_lemmas) < 2:
        return 0.0

    cohesive_count = 0

    for i in range(len(sent_lemmas) - 1):
        for j in range(i + 1, len(sent_lemmas)):
            li = sent_lemmas[i]
            lj = sent_lemmas[j]

            # 1) Lexical repetition: shared lemmas
            shared = li & lj
            cohesive_count += len(shared)

            # 2) Semantic relations via WordNet
            for lemma in li:
                related = get_related_lemmas(lemma)
                cohesive_count += len(related & lj)

    return float(cohesive_count) / float(m)


def sentence_vector(sent, vector_size):
    """
    Represent a sentence as the average of token vectors.
    If no token has a vector, return a zero vector.
    """
    vecs = [
        tok.vector
        for tok in sent
        if tok.has_vector and not tok.is_punct and not tok.is_space
    ]
    if not vecs:
        return np.zeros(vector_size, dtype="float32")
    return np.mean(vecs, axis=0)


def coherence_single(text, nlp):
    """
    Compute Coherence (CoH) for a single document as the average
    cosine similarity between adjacent sentence vectors:

        CoH = (1 / (k-1)) * sum_{i=1}^{k-1} cos(h_i, h_{i+1})

    where h_i is the sentence/topic vector for sentence i.

    If the document has fewer than 2 sentences, CoH = 0.0.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    if nlp.vocab.vectors_length == 0:
        raise ValueError(
            "The loaded spaCy model does not contain word vectors "
            "(nlp.vocab.vectors_length == 0). "
            "Use a model like 'en_core_web_md' or similar."
        )

    doc = nlp(text)
    sentences = list(doc.sents)
    k = len(sentences)

    if k < 2:
        # Only one sentence: no adjacent pair, coherence = 0.0
        return 0.0

    vector_size = nlp.vocab.vectors_length
    sent_vectors = [sentence_vector(sent, vector_size) for sent in sentences]

    sims = []
    for i in range(k - 1):
        v1 = sent_vectors[i]
        v2 = sent_vectors[i + 1]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        denom = norm1 * norm2
        if denom == 0.0:
            # Skip pairs where at least one sentence vector is zero
            continue
        cos_sim = float(np.dot(v1, v2) / denom)
        sims.append(cos_sim)

    if not sims:
        return 0.0

    return float(np.mean(sims))


def compute_lexical_cohesion_vector(df, nlp, column="text"):
    """
    Compute LC for each row of a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the texts.
    nlp : spaCy Language object
        Pre-loaded spaCy pipeline with lemmatizer, POS tagger, etc.
    column : str, default "text"
        Name of the column that contains the text.

    Returns
    -------
    np.ndarray
        1D array of LC scores, length == len(df).
    """
    texts = df[column].fillna("").astype(str)
    scores = [lexical_cohesion_single(t, nlp) for t in texts]
    return np.array(scores, dtype="float32")


def compute_coherence_vector(df, nlp, column="text"):
    """
    Compute CoH for each row of a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the texts.
    nlp : spaCy Language object
        Pre-loaded spaCy pipeline with word vectors.
    column : str, default "text"
        Name of the column that contains the text.

    Returns
    -------
    np.ndarray
        1D array of CoH scores, length == len(df).
    """
    texts = df[column].fillna("").astype(str)
    scores = [coherence_single(t, nlp) for t in texts]
    return np.array(scores, dtype="float32")


def compute_discourse_measures(df, nlp, column="text"):
    """
    Compute both LC and CoH for each row of a DataFrame and return
    them in a dictionary.

    Returns
    -------
    dict
        {
            "LC":  np.ndarray of lexical cohesion scores,
            "CoH": np.ndarray of coherence scores
        }
    """
    lc_vec = compute_lexical_cohesion_vector(df, nlp, column=column)
    coh_vec = compute_coherence_vector(df, nlp, column=column)
    return {"LC": lc_vec, "CoH": coh_vec}
