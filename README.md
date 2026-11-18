# Support Response Generator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/nebx8mzhzt/badge.svg)](https://ask.thili.ai/b/nebx8mzhzt)

The Support Response Generator is a CLI tool designed for retail customer support agents. It leverages next-word prediction to help agents quickly draft responses to common queries, improving efficiency and consistency in customer interactions.

## What you are building

**Goal.** You will build a Command Line Interface (CLI) tool that helps customer support agents quickly draft responses to common queries by generating next-word suggestions based on a corpus of past support interactions.

**Done when.** The tool successfully generates next-word suggestions for support agents, using an n-gram model trained on a prepared corpus of support interactions, accessible via a command-line interface.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare an anonymized corpus of customer support interactions.**  
      Hint: Focus on text cleaning and tokenization for the corpus.
- [ ] **2. Implement the n-gram model with smoothing.**  
      Hint: Build the n-gram model and apply smoothing techniques as learned in `lfllm-lfnextword`.
- [ ] **3. Create a text generation function to suggest response continuations.**  
      Hint: Implement the next-word prediction logic to generate continuations.
- [ ] **4. Build a command-line interface for agents to get suggestions.**  
      Hint: Design a CLI to interact with the text generation function.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Implement additional smoothing techniques for the n-gram model to compare performance.
- Allow the CLI to display multiple next-word suggestions and let the user choose.
