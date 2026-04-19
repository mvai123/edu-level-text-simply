"""Rule-based simplification baseline for the three target reading levels."""

import argparse
import csv
import json
import logging
import re
from typing import Optional

import nltk
import spacy
import textstat
from nltk.corpus import wordnet
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

for _pkg in ["punkt", "averaged_perceptron_tagger", "wordnet", "stopwords", "punkt_tab"]:
    nltk.download(_pkg, quiet=True)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise SystemExit(
        "spaCy model not found. Run:\n  python -m spacy download en_core_web_sm"
    )

# Flesch-Kincaid targets used as a rough guardrail for each level.
LEVEL_TARGETS = {
    "kindergarten": {"max_fkgl": 1.0, "max_words_per_sent": 8, "max_syllables_per_word": 1},
    "primary": {"max_fkgl": 5.0, "max_words_per_sent": 15, "max_syllables_per_word": 2},
    "secondary": {"max_fkgl": 9.0, "max_words_per_sent": 20, "max_syllables_per_word": 3},
}

# Small hand-built vocabulary. WordNet fills in a few extra substitutions below.
SIMPLE_VOCAB: dict[str, dict[str, str]] = {
    # Academic / formal -> plain
    "utilize":       {"kindergarten": "use",       "primary": "use",        "secondary": "use"},
    "obtain":        {"kindergarten": "get",        "primary": "get",        "secondary": "get"},
    "sufficient":    {"kindergarten": "enough",     "primary": "enough",     "secondary": "enough"},
    "require":       {"kindergarten": "need",       "primary": "need",       "secondary": "need"},
    "demonstrate":   {"kindergarten": "show",       "primary": "show",       "secondary": "show"},
    "commence":      {"kindergarten": "start",      "primary": "start",      "secondary": "start"},
    "terminate":     {"kindergarten": "end",        "primary": "end",        "secondary": "end"},
    "endeavour":     {"kindergarten": "try",        "primary": "try",        "secondary": "try"},
    "endeavor":      {"kindergarten": "try",        "primary": "try",        "secondary": "try"},
    "purchase":      {"kindergarten": "buy",        "primary": "buy",        "secondary": "buy"},
    "indicate":      {"kindergarten": "show",       "primary": "show",       "secondary": "show"},
    "assistance":    {"kindergarten": "help",       "primary": "help",       "secondary": "help"},
    "approximately": {"kindergarten": "about",      "primary": "about",      "secondary": "around"},
    "however":       {"kindergarten": "but",        "primary": "but",        "secondary": "however"},
    "therefore":     {"kindergarten": "so",         "primary": "so",         "secondary": "therefore"},
    "nevertheless":  {"kindergarten": "but",        "primary": "still",      "secondary": "however"},
    "subsequently":  {"kindergarten": "then",       "primary": "then",       "secondary": "after that"},
    "immediately":   {"kindergarten": "right away", "primary": "right away", "secondary": "at once"},
    "numerous":      {"kindergarten": "many",       "primary": "many",       "secondary": "many"},
    "frequently":    {"kindergarten": "often",      "primary": "often",      "secondary": "often"},
    "difficult":     {"kindergarten": "hard",       "primary": "hard",       "secondary": "difficult"},
    "complicated":   {"kindergarten": "hard",       "primary": "tricky",     "secondary": "complex"},
    "important":     {"kindergarten": "big",        "primary": "important",  "secondary": "important"},
    "additional":    {"kindergarten": "more",       "primary": "more",       "secondary": "extra"},
    "construct":     {"kindergarten": "build",      "primary": "build",      "secondary": "build"},
    "comprehend":    {"kindergarten": "understand", "primary": "understand", "secondary": "understand"},
    "fortunate":     {"kindergarten": "lucky",      "primary": "lucky",      "secondary": "lucky"},
    "residence":     {"kindergarten": "home",       "primary": "home",       "secondary": "house"},
    "physician":     {"kindergarten": "doctor",     "primary": "doctor",     "secondary": "doctor"},
    "automobile":    {"kindergarten": "car",        "primary": "car",        "secondary": "car"},
    "children":      {"kindergarten": "kids",       "primary": "children",   "secondary": "children"},
}


def compute_readability(text: str) -> dict:
    """Return the readability numbers we use while debugging the pipeline."""
    return {
        "fkgl": textstat.flesch_kincaid_grade(text),
        "flesch_ease": textstat.flesch_reading_ease(text),
        "avg_sent_len": textstat.avg_sentence_length(text),
        "avg_syllables": textstat.avg_syllables_per_word(text),
    }


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    return nltk.sent_tokenize(text)


def _wordnet_simpler(word: str, level: str) -> Optional[str]:
    """Look for a shorter WordNet synonym."""
    synsets = wordnet.synsets(word)
    if not synsets:
        return None

    candidates = []
    for syn in synsets[:3]:
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower() and len(name) < len(word):
                candidates.append(name)

    if not candidates:
        return None

    return min(candidates, key=len)


def lexical_simplify(sentence: str, level: str) -> str:
    """Swap known difficult words for easier alternatives."""
    doc = nlp(sentence)
    new_tokens = []

    for token in doc:
        word_lower = token.lower_
        if word_lower in SIMPLE_VOCAB and level in SIMPLE_VOCAB[word_lower]:
            replacement = SIMPLE_VOCAB[word_lower][level]
            if token.text[0].isupper():
                replacement = replacement.capitalize()
            new_tokens.append(replacement)
        elif (
            level in ("kindergarten", "primary")
            and token.pos_ in ("NOUN", "VERB", "ADJ", "ADV")
            and textstat.syllable_count(token.text) > LEVEL_TARGETS[level]["max_syllables_per_word"]
        ):
            simpler = _wordnet_simpler(word_lower, level)
            if simpler:
                if token.text[0].isupper():
                    simpler = simpler.capitalize()
                new_tokens.append(simpler)
            else:
                new_tokens.append(token.text_with_ws.rstrip())
        else:
            new_tokens.append(token.text_with_ws.rstrip())

    result = ""
    for token, new_tok in zip(doc, new_tokens):
        result += new_tok + token.whitespace_
    return result.strip()


_CLAUSE_TAGS = {
    "advcl",    # adverbial clause
    "relcl",    # relative clause
    "ccomp",    # clausal complement
    "xcomp",    # open clausal complement (keep for primary/secondary)
    "acl",      # adjectival clause
}

_INTRO_PHRASES = re.compile(
    r"^(In other words,|For example,|For instance,|That is,|"
    r"In fact,|As a result,|On the other hand,|In contrast,|"
    r"In addition,|Furthermore,|Moreover,|However,|Therefore,|"
    r"Nevertheless,|In conclusion,)\s+",
    re.IGNORECASE,
)


def remove_clauses(sentence: str, level: str) -> str:
    """Drop some dependent clauses for the two younger reading levels."""
    if level == "secondary":
        return sentence

    doc = nlp(sentence)
    remove_deps = _CLAUSE_TAGS if level == "kindergarten" else {"advcl", "relcl", "acl"}

    remove_indices: set[int] = set()
    for token in doc:
        if token.dep_ in remove_deps:
            subtree_indices = {t.i for t in token.subtree}
            if len(subtree_indices) <= 3 and level == "primary":
                continue
            remove_indices.update(subtree_indices)

    if not remove_indices:
        return sentence

    kept = [
        token.text + token.whitespace_
        for token in doc
        if token.i not in remove_indices
    ]
    result = "".join(kept).strip()

    result = re.sub(r"\s+,", ",", result)
    result = re.sub(r"\s+\.", ".", result)
    result = re.sub(r",\s*\.", ".", result)
    result = re.sub(r"\s{2,}", " ", result)
    result = _INTRO_PHRASES.sub("", result)

    return result if result.strip() else sentence


def passive_to_active(sentence: str) -> str:
    """Convert straightforward passives like "X was seen by Y" to active voice."""
    doc = nlp(sentence)

    for token in doc:
        if token.dep_ == "nsubjpass":
            passive_subject = token
            verb = token.head

            agent = None
            for child in verb.children:
                if child.dep_ == "agent":
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            agent = grandchild
                            break

            if agent is None:
                continue

            subj_span = doc[passive_subject.left_edge.i : passive_subject.right_edge.i + 1]
            agent_span = doc[agent.left_edge.i : agent.right_edge.i + 1]
            main_verb_lemma = verb.lemma_

            new_sentence = (
                agent_span.text.capitalize()
                + " "
                + main_verb_lemma
                + "s "
                + subj_span.text.lower()
                + "."
            )
            return new_sentence

    return sentence

def trim_long_sentence(sentence: str, max_words: int) -> str:
    """Shorten a sentence by cutting it at a word boundary."""
    words = sentence.split()
    if len(words) <= max_words:
        return sentence

    truncated = " ".join(words[:max_words])
    if not truncated.endswith((".", "!", "?")):
        truncated = truncated.rstrip(",;:") + "."
    return truncated


def simplify_for_level(text: str, level: str) -> str:
    """Run the rule-based simplifier for one target level."""
    target = LEVEL_TARGETS[level]
    initial = compute_readability(text)
    logger.debug("Initial FKGL=%.1f, ease=%.1f", initial["fkgl"], initial["flesch_ease"])

    sentences = split_sentences(text)
    simplified_sentences = []

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        sent = lexical_simplify(sent, level)
        sent = remove_clauses(sent, level)
        sent = passive_to_active(sent)

        if level == "kindergarten":
            sent = trim_long_sentence(sent, target["max_words_per_sent"])

        if sent:
            simplified_sentences.append(sent)

    result = " ".join(simplified_sentences)

    final = compute_readability(result)
    logger.debug(
        "Final [%s] FKGL=%.1f, ease=%.1f",
        level, final["fkgl"], final["flesch_ease"],
    )

    return result


def simplify_text(original: str) -> dict[str, str]:
    """Return simplified versions for all three levels."""
    return {
        "original_text": original,
        "simplified_kindergarten": simplify_for_level(original, "kindergarten"),
        "simplified_primary_school": simplify_for_level(original, "primary"),
        "simplified_secondary_school": simplify_for_level(original, "secondary"),
    }


def read_jsonl(path: str, text_col: str) -> list[str]:
    """Read texts from a JSONL column."""
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping line %d (JSON error): %s", lineno, exc)
                continue

            if text_col not in record:
                logger.warning(
                    "Line %d missing column '%s'. Available: %s",
                    lineno, text_col, list(record.keys()),
                )
                continue

            texts.append(str(record[text_col]))

    logger.info("Loaded %d records from %s", len(texts), path)
    return texts


def write_csv(rows: list[dict], path: str) -> None:
    """Write simplification rows to CSV."""
    if not rows:
        logger.warning("No rows to write.")
        return

    fieldnames = ["original_text", "simplified_kindergarten", "simplified_primary_school", "simplified_secondary_school"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved %d rows to %s", len(rows), path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Text simplification pipeline for Kindergarten / Primary / Secondary levels."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to input .jsonl file."
    )
    parser.add_argument(
        "--output", "-o", default="simplified.csv",
        help="Path to output .csv file (default: simplified.csv)."
    )
    parser.add_argument(
        "--text_col", default="original_text",
        help="Column name in JSONL that contains the source text (default: original_text)."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N records (useful for testing)."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    texts = read_jsonl(args.input, args.text_col)
    if args.limit:
        texts = texts[: args.limit]
        logger.info("Limiting to first %d records.", args.limit)

    results = []
    for text in tqdm(texts, desc="Simplifying", unit="doc"):
        try:
            row = simplify_text(text)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error processing text: %s", exc)
            row = {
                "original_text": text,
                "simplified_kindergarten": "",
                "simplified_primary_school": "",
                "simplified_secondary_school": "",
            }
        results.append(row)

    write_csv(results, args.output)
    print(f"\nDone. Output saved to: {args.output}")


if __name__ == "__main__":
    main()
