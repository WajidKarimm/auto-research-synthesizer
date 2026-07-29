"""Command-line entrypoint for `python -m ars`."""

import sys

from dotenv import load_dotenv

from ars.core.graph import _print_sources, run


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    load_dotenv()

    if not args:
        print('Usage: python -m ars "your research question"')
        return 1

    question = " ".join(args)
    print(f"\nResearch question: {question}\n{'-' * 60}")
    final_state = run(question)
    print(f"\n{'-' * 60}\nAnswer:\n{final_state['answer']}\n")
    print(f"{'-' * 60}")
    if final_state.get("sources"):
        _print_sources(final_state["sources"])
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
