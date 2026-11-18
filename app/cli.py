"""Command-line entry point: `python -m app.cli --help`."""
from __future__ import annotations

import argparse
import sys

from app import core


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--category", "-c", help="train only on replies from this category (e.g. refund)")
    p.add_argument("--n", type=int, default=3, help="n-gram order (default 3)")
    p.add_argument("--k", type=float, default=0.01, help="add-k smoothing constant (default 0.01, best on evaluate)")
    p.add_argument("--smoothing", choices=["add_k", "interpolated"], default="add_k")


def _model(args) -> core.NGramModel:
    return core.train(category=args.category, n=args.n, k=args.k, smoothing=args.smoothing)


def interactive(model: core.NGramModel, top_k: int = 5, inp=input, out=print) -> str:
    """Build a reply word by word: pick a numbered suggestion, type your own words, or finish."""
    out("Type words to add them, a number to pick a suggestion, 'g' to auto-complete, "
        "'u' to undo, empty line to finish.")
    tokens: list[str] = []
    while True:
        text = core.detokenize(tokens)
        options = [w for w, _ in core.suggest_next(model, text, top_k)]
        out(f"\nDraft: {text or '(empty)'}")
        out("  " + "   ".join(f"[{i}] {'(end)' if w == core.EOS else w}" for i, w in enumerate(options, 1)))
        try:
            choice = inp("> ").strip()
        except EOFError:
            choice = ""
        if not choice:
            break
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            word = options[int(choice) - 1]
            if word == core.EOS:
                break
            tokens.append(word)
        elif choice == "g":
            tokens = core.tokenize(core.generate(model, text))
            break
        elif choice == "u":
            tokens = tokens[:-1]
        else:
            tokens.extend(core.tokenize(choice))
    final = core.detokenize(tokens)
    out(f"\nFinal reply: {final}")
    return final


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="support-response-generator", description="Customer support agents use this CLI tool to quickly draft responses to common queries by generating next-word suggestions based on a corpus of past support interactions.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")

    sub.add_parser("prepare", help="anonymise + clean corpus/raw_support_logs.tsv into corpus/support_corpus.tsv")
    sub.add_parser("categories", help="list the reply categories in the corpus")

    p = sub.add_parser("suggest", help="show the most likely next words for a partial reply")
    p.add_argument("text", help='partial reply, e.g. "thank you for"')
    p.add_argument("--top", type=int, default=5, help="how many suggestions (default 5)")
    _add_model_args(p)

    p = sub.add_parser("complete", help="draft the rest of a reply")
    p.add_argument("text", nargs="?", default="", help="start of the reply (optional)")
    p.add_argument("--max-words", type=int, default=40)
    p.add_argument("--sample", action="store_true", help="sample for varied drafts instead of greedy")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--drafts", type=int, default=1, help="how many drafts (use with --sample)")
    _add_model_args(p)

    p = sub.add_parser("interactive", help="build a reply word by word, choosing from suggestions")
    p.add_argument("--top", type=int, default=5)
    _add_model_args(p)

    sub.add_parser("evaluate", help="compare smoothing settings by perplexity on held-out replies")

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Support Response Generator is ready. Open README.md for your milestones.")
    elif args.command == "prepare":
        rows = core.prepare_corpus()
        print(f"Prepared {rows} anonymised interactions -> {core.CLEAN_CORPUS.relative_to(core.ROOT)}")
    elif args.command == "categories":
        print("\n".join(core.categories()))
    elif args.command == "suggest":
        model = _model(args)
        for i, (word, p) in enumerate(core.suggest_next(model, args.text, args.top), 1):
            print(f"{i}. {'(end of reply)' if word == core.EOS else word:<20} {p:.3f}")
    elif args.command == "complete":
        model = _model(args)
        for i in range(args.drafts):
            seed = None if args.seed is None else args.seed + i
            draft = core.generate(model, args.text, args.max_words, args.sample, args.temperature, seed)
            print(f"{i + 1}. {draft}" if args.drafts > 1 else draft)
    elif args.command == "interactive":
        interactive(_model(args), args.top)
    elif args.command == "evaluate":
        print(f"{'n':>2}  {'k':>5}  {'smoothing':<13} perplexity")
        for r in core.evaluate():
            print(f"{r['n']:>2}  {r['k']:>5}  {r['smoothing']:<13} {r['perplexity']}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
