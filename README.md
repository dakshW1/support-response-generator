# Support Response Generator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/nebx8mzhzt/badge.svg)](https://ask.thili.ai/b/nebx8mzhzt)
[![tests](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

A command-line tool for retail customer-support agents. It suggests the next word, or drafts the rest of a reply,
using an **n-gram language model trained on past support replies** - the same idea as phone-keyboard autocomplete,
built from word counts with no neural network and no API keys.

```text
$ python -m app.cli suggest "thank you for"
1. reaching             0.341
2. your                 0.102
3. contacting           0.022
...

$ python -m app.cli complete "i am sorry"
I am sorry for the trouble. Please check that the item within 30 days of delivery. ...

$ python -m app.cli complete "thank you for reaching out." -c account
Thank you for reaching out. You can reset your password from the account settings page.
```

## Milestones

- [x] **1. Prepare an anonymized corpus of customer support interactions.**
      `corpus/raw_support_logs.tsv` (51 interactions across 7 categories) is anonymised, cleaned and tokenised into
      `corpus/support_corpus.tsv`.
- [x] **2. Implement the n-gram model with smoothing.** `NGramModel` with add-k (Laplace) smoothing + back-off,
      and interpolated smoothing as a second method to compare.
- [x] **3. Create a text generation function to suggest response continuations.** `suggest_next` (top-k next words)
      and `generate` (greedy or temperature sampling).
- [x] **4. Build a command-line interface for agents to get suggestions.** `suggest`, `complete`, `interactive`,
      `evaluate`, `prepare`, `categories`.

Stretch ideas done: a second smoothing technique compared by perplexity, and an interactive mode where the agent
picks from numbered suggestions.

## Run it

You need Python 3.12+ and git. No model, API key or `.env` is needed - everything runs locally from word counts.

```bash
python3.12 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m app.cli prepare                                   # anonymise + clean the corpus
python -m app.cli suggest "your refund has been"            # next-word suggestions
python -m app.cli complete "i am sorry" -c shipping         # draft a full reply
python -m app.cli complete "your order" --sample --drafts 3 --seed 1   # several varied drafts
python -m app.cli interactive -c returns                    # build a reply word by word
python -m app.cli evaluate                                  # compare smoothing settings
pytest
```

Every model command accepts `--category/-c` (`account, general, payment, product, refund, returns, shipping`),
`--n` (n-gram order), `--k` (smoothing constant) and `--smoothing add_k|interpolated`.

## How it works

**1. Corpus preparation** (`app/core.py`: `anonymize`, `clean`, `tokenize`, `prepare_corpus`)
- The raw logs are fictional but realistic, and deliberately contain personal details.
- `anonymize` replaces emails with `<email>`, phone numbers (`+91 98450 12345`, `9845012345`) with `<phone>`, and
  order ids (`#A48213`, long numbers) with `<order_id>`. Emails are masked first because they can contain digits.
- Text is lower-cased and whitespace-normalised, then split into words and punctuation (`.` `,` `!` `?`), so the
  model learns where sentences end. Each reply is wrapped in `<s> ... </s>`.

**2. The n-gram model** (`NGramModel`)
- `fit` counts every n-gram (and every lower order) in the replies.
- Maximum likelihood: `P(w | ctx) = C(ctx, w) / C(ctx)` - but any unseen word gets probability 0, and one zero
  makes a whole reply impossible (infinite perplexity).
- **Add-k smoothing**: `P(w | ctx) = (C(ctx, w) + k) / (C(ctx) + k·V)`. If the whole context was never seen, the
  model backs off to the longest shorter context it has seen.
- **Interpolated smoothing**: a weighted mix of trigram, bigram and unigram add-k estimates.
- Unknown words map to `<unk>`. Tests check every distribution sums to 1.

**3. Generation** (`suggest_next`, `generate`)
- `suggest_next` ranks the whole vocabulary by `P(w | last n-1 words)`.
- `generate` is greedy by default (repeatable drafts) and blocks any repeated 4-gram, which otherwise makes greedy
  decoding loop. `--sample` draws from the top 5 words with a temperature. Sampling from the full smoothed
  distribution gave gibberish: with ~250 words, the tiny `k` mass on every unseen word adds up to most of the
  probability.

**4. CLI** (`app/cli.py`) - argparse sub-commands wrapping the functions above.

## Evaluation

`python -m app.cli evaluate` holds out every 5th reply and measures **perplexity** (`exp` of the average negative
log-probability per token; lower is better):

| n | k    | smoothing    | perplexity |
|---|------|--------------|-----------:|
| 1 | 1.0  | add-k        | 119.21 |
| 2 | 1.0  | add-k        | 69.33  |
| 2 | 0.1  | add-k        | 26.66  |
| 3 | 1.0  | add-k        | 80.28  |
| 3 | 0.1  | add-k        | 27.67  |
| 3 | 0.01 | add-k        | **15.43** |
| 3 | 0.1  | interpolated | 29.63  |
| 4 | 0.1  | interpolated | 34.12  |

What this shows:
- More context helps (unigram 119 -> bigram/trigram ~27) because support replies are very formulaic.
- With a vocabulary this small, **k = 1 over-smooths badly**: it hands too much probability to words that never
  follow the context. Trigram with k = 1 is even worse than bigram, because trigram contexts are rarer, so the
  added `k·V` dominates the real counts.
- Small k (0.01) with back-off works best, so it is the default.
- The held-out set is only ~10 replies, so these numbers are a rough comparison, not a benchmark.

## Limitations and next steps

- 51 interactions is a tiny corpus. More real (anonymised) logs would improve suggestions most.
- The model only uses the agent's own words. Conditioning on the customer's message (or its category, detected
  automatically) would make drafts more relevant.
- Kneser-Ney smoothing would likely beat add-k.

## Project layout

```text
app/core.py        corpus prep, NGramModel, suggestion + generation, evaluation
app/cli.py         command-line interface
corpus/            raw (fictional) logs and the anonymised, cleaned corpus
tests/             one test block per milestone + the starter smoke tests
app/llm.py, app/storage.py   starter plumbing (not used by the n-gram model)
```

Built for the askThili drill *Build a Next-Word Predictor from Scratch*. Written with help from an AI coding
assistant (Claude Code).
