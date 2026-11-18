"""Support Response Generator: an n-gram language model over past support replies.

Built in four milestones (see README.md):
  1. corpus preparation  - anonymise, clean and tokenise past support interactions
  2. n-gram model        - counts -> probabilities, with add-k and interpolated smoothing
  3. generation          - next-word suggestions and full reply continuations
  4. CLI                 - lives in app/cli.py and calls the functions below
"""
from __future__ import annotations

import csv
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

BOS, EOS, UNK = "<s>", "</s>", "<unk>"
PLACEHOLDERS = ("<email>", "<phone>", "<order_id>")

ROOT = Path(__file__).resolve().parent.parent
RAW_CORPUS = ROOT / "corpus" / "raw_support_logs.tsv"
CLEAN_CORPUS = ROOT / "corpus" / "support_corpus.tsv"


# --- Milestone 1: Prepare an anonymized corpus of customer support interactions.

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_ORDER_HASH = re.compile(r"#\s?[A-Za-z]?\d{4,}")
_PHONE = re.compile(r"(?:\+?\d{1,3}[\s-]?)?\d{5}[\s-]?\d{5}\b")
_LONG_NUMBER = re.compile(r"\b\d{6,}\b")
_TOKEN = re.compile(r"<email>|<phone>|<order_id>|[a-z0-9]+(?:'[a-z]+)?|[.,!?]")


def anonymize(text: str) -> str:
    """Replace personal details with placeholders so no customer can be identified.

    Order matters: emails first (they contain digits), then '#'-style order ids,
    then phone numbers, then any remaining long number (treated as an order id).
    """
    text = _EMAIL.sub("<email>", text)
    text = _ORDER_HASH.sub("<order_id>", text)
    text = _PHONE.sub("<phone>", text)
    text = _LONG_NUMBER.sub("<order_id>", text)
    return text


def clean(text: str) -> str:
    """Lower-case and normalise whitespace and quote characters."""
    text = text.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text: str) -> list[str]:
    """Split cleaned text into word and punctuation tokens. Placeholders stay whole."""
    return _TOKEN.findall(clean(text))


def detokenize(tokens: list[str]) -> str:
    """Join tokens back into readable text (no space before punctuation)."""
    out = ""
    for tok in tokens:
        if tok in (BOS, EOS):
            continue
        out += tok if tok in ".,!?" or not out else " " + tok
    # capitalise the start of each sentence
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), out)


def prepare_corpus(raw_path: Path = RAW_CORPUS, out_path: Path = CLEAN_CORPUS) -> int:
    """Read raw support logs, anonymise + clean both sides, write the clean corpus. Returns row count."""
    with open(raw_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["category", "customer", "agent"])
        for row in rows:
            writer.writerow([row["category"].strip().lower(),
                             clean(anonymize(row["customer"])),
                             clean(anonymize(row["agent"]))])
    return len(rows)


def load_replies(path: Path = CLEAN_CORPUS, category: str | None = None) -> list[list[str]]:
    """Load agent replies as token lists, optionally only one category. Prepares the corpus if missing."""
    if not path.exists():
        prepare_corpus(out_path=path)
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if category:
        rows = [r for r in rows if r["category"] == category.lower()]
    return [tokenize(r["agent"]) for r in rows]


def categories(path: Path = CLEAN_CORPUS) -> list[str]:
    if not path.exists():
        prepare_corpus(out_path=path)
    with open(path, encoding="utf-8", newline="") as f:
        return sorted({r["category"] for r in csv.DictReader(f, delimiter="\t")})


# --- Milestone 2: Implement the n-gram model with smoothing.

class NGramModel:
    """Word n-gram language model.

    smoothing="add_k":        P(w | ctx) = (C(ctx, w) + k) / (C(ctx) + k * V)
                              If the full context was never seen, back off to the
                              longest shorter context that was seen.
    smoothing="interpolated": P(w | ctx) = sum_i lambda_i * P_addk(w | last i words of ctx)
                              mixing every order from n down to unigrams.
    """

    def __init__(self, n: int = 3, k: float = 0.01, smoothing: str = "add_k",
                 lambdas: tuple[float, ...] | None = None):
        if n < 1:
            raise ValueError("n must be at least 1")
        if smoothing not in ("add_k", "interpolated"):
            raise ValueError("smoothing must be 'add_k' or 'interpolated'")
        self.n, self.k, self.smoothing = n, k, smoothing
        weights = lambdas or tuple(2.0 ** -i for i in range(n))  # highest order weighted most
        if len(weights) != n:
            raise ValueError("need one lambda per order")
        total = sum(weights)
        self.lambdas = tuple(w / total for w in weights)
        # counts[order][context_tuple][word]; order 1 uses the empty context ()
        self.counts: dict[int, dict[tuple, Counter]] = {o: defaultdict(Counter) for o in range(1, n + 1)}
        self.vocab: set[str] = set()

    def fit(self, sentences: list[list[str]]) -> "NGramModel":
        word_freq = Counter(w for s in sentences for w in s)
        self.vocab = set(word_freq) | {EOS, UNK}
        for sent in sentences:
            padded = [BOS] * (self.n - 1) + sent + [EOS]
            for i in range(self.n - 1, len(padded)):
                word = padded[i]
                for order in range(1, self.n + 1):
                    ctx = tuple(padded[i - order + 1:i])
                    self.counts[order][ctx][word] += 1
        return self

    @property
    def V(self) -> int:
        return len(self.vocab)

    def _norm(self, word: str) -> str:
        return word if word in self.vocab or word == BOS else UNK

    def _context(self, history: list[str]) -> tuple:
        padded = [BOS] * (self.n - 1) + [self._norm(w) for w in history]
        return tuple(padded[len(padded) - (self.n - 1):]) if self.n > 1 else ()

    def _addk(self, word: str, ctx: tuple) -> float:
        table = self.counts[len(ctx) + 1].get(ctx)
        seen = table[word] if table else 0
        total = sum(table.values()) if table else 0
        return (seen + self.k) / (total + self.k * self.V)

    def prob(self, word: str, history: list[str]) -> float:
        """P(word | the last n-1 words of history)."""
        word, ctx = self._norm(word), self._context(history)
        if self.smoothing == "interpolated":
            return sum(lam * self._addk(word, ctx[len(ctx) - (order - 1):] if order > 1 else ())
                       for lam, order in zip(self.lambdas, range(self.n, 0, -1)))
        # add-k with back-off to the longest context we have actually seen
        while ctx and ctx not in self.counts[len(ctx) + 1]:
            ctx = ctx[1:]
        return self._addk(word, ctx)

    def distribution(self, history: list[str]) -> dict[str, float]:
        """Probability of every vocabulary word (except <unk>) following history."""
        return {w: self.prob(w, history) for w in self.vocab if w != UNK}

    def perplexity(self, sentences: list[list[str]]) -> float:
        """exp of the average negative log-probability per token (including </s>). Lower is better."""
        log_sum, count = 0.0, 0
        for sent in sentences:
            history: list[str] = []
            for word in sent + [EOS]:
                log_sum += -math.log(self.prob(word, history))
                history.append(word)
                count += 1
        return math.exp(log_sum / count) if count else float("inf")


def train(category: str | None = None, n: int = 3, k: float = 0.01,
          smoothing: str = "add_k", path: Path = CLEAN_CORPUS) -> NGramModel:
    """Train a model on all replies, or on one category (falls back to all if it is empty)."""
    replies = load_replies(path, category) or load_replies(path)
    return NGramModel(n=n, k=k, smoothing=smoothing).fit(replies)


def evaluate(path: Path = CLEAN_CORPUS, test_every: int = 5,
             settings: list[tuple[int, float, str]] | None = None) -> list[dict]:
    """Hold out every `test_every`-th reply and report perplexity for several model settings."""
    replies = load_replies(path)
    train_set = [r for i, r in enumerate(replies) if i % test_every]
    test_set = [r for i, r in enumerate(replies) if not i % test_every]
    settings = settings or [(1, 1.0, "add_k"), (2, 1.0, "add_k"), (2, 0.1, "add_k"),
                            (3, 1.0, "add_k"), (3, 0.1, "add_k"), (3, 0.01, "add_k"),
                            (3, 0.1, "interpolated"), (4, 0.1, "interpolated")]
    results = []
    for n, k, smoothing in settings:
        model = NGramModel(n=n, k=k, smoothing=smoothing).fit(train_set)
        results.append({"n": n, "k": k, "smoothing": smoothing,
                        "perplexity": round(model.perplexity(test_set), 2)})
    return results


# --- Milestone 3: Create a text generation function to suggest response continuations.

def suggest_next(model: NGramModel, prefix: str, top_k: int = 5) -> list[tuple[str, float]]:
    """The top_k most likely next words after prefix, with their probabilities."""
    dist = model.distribution(tokenize(prefix))
    return sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]


def _repeats(tokens: list[str], word: str, size: int) -> bool:
    """True if appending word would repeat an n-gram of this size already in tokens."""
    cand = tuple(tokens[-(size - 1):] + [word]) if size > 1 else (word,)
    if len(cand) < size:
        return False
    return any(tuple(tokens[i:i + size]) == cand for i in range(len(tokens) - size + 1))


def generate(model: NGramModel, prefix: str = "", max_words: int = 40, sample: bool = False,
             temperature: float = 0.7, seed: int | None = None, top_k: int = 5,
             no_repeat: int = 4) -> str:
    """Continue prefix until </s> or max_words.

    sample=False: greedy (always the most likely word) - stable, repeatable drafts.
    sample=True:  sample among the top_k words, sharpened by temperature - varied drafts.
                  Restricting to top_k matters: smoothing gives every unseen word a little
                  mass, and over the whole vocabulary that adds up to most of the probability.
    no_repeat:    never repeat an n-gram of this size, which stops greedy decoding from looping.
    """
    rng = random.Random(seed)
    tokens = tokenize(prefix)
    for _ in range(max_words):
        ranked = sorted(model.distribution(tokens).items(), key=lambda kv: (-kv[1], kv[0] != EOS))
        ranked = [(w, p) for w, p in ranked if w == EOS or not _repeats(tokens, w, no_repeat)]
        if not ranked:
            break
        if sample:
            pool = ranked[:max(top_k, 1)]
            weights = [p ** (1.0 / max(temperature, 1e-6)) for _, p in pool]
            nxt = rng.choices([w for w, _ in pool], weights=weights, k=1)[0]
        else:
            nxt = ranked[0][0]
        if nxt == EOS:
            break
        tokens.append(nxt)
    return detokenize(tokens)


# --- Milestone 4: Build a command-line interface for agents to get suggestions.
# The CLI lives in app/cli.py and calls the functions above.
