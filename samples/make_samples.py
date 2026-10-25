"""Generates samples/sample_texts.csv using Python's csv module so quoting,
embedded newlines and non-ASCII characters (CJK, emoji, ...) are all handled
correctly.
"""

import csv
from pathlib import Path

ROWS = [
    (
        "english prose",
        "Tokenization is the process of splitting text into smaller units called "
        "tokens. These tokens are the basic building blocks that large language "
        "models read and generate.",
    ),
    (
        "malay prose",
        "Pemprosesan bahasa semula jadi membolehkan komputer memahami bahasa "
        "manusia dengan lebih baik. Model bahasa besar semakin digunakan dalam "
        "pelbagai aplikasi harian.",
    ),
    (
        "japanese text",
        "自然言語処理は人工知能の重要な分野です。コンピュータが人間の言葉を理解し、翻訳や要約などのタスクを自動化できます。",
    ),
    (
        "python code",
        "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
    ),
    (
        "numbers dates",
        "The meeting is scheduled for 2025-03-15 at 14:30. Invoice #0042 totals "
        "$1,234.56 with a 7.5% tax rate, and 3.14159 is the ratio of circumference "
        "to diameter.",
    ),
    (
        "short sentence",
        "Hello world.",
    ),
    (
        "long paragraph",
        "Large language models have transformed the way we interact with computers. "
        "At their core, they operate on sequences of tokens rather than raw "
        "characters, which means the same sentence can be represented very "
        "differently depending on the tokenizer. Understanding these differences "
        "is essential for engineers who want to control costs, respect context "
        "windows, and design robust prompts. This dashboard compares the output "
        "of three popular tokenizers side by side so that the impact of language, "
        "code, and formatting choices on token counts becomes immediately visible.",
    ),
    (
        "french text",
        "L'intelligence artificielle transforme notre façon de travailler et de "
        "communiquer. Les modèles de langage apprennent à partir de grandes "
        "quantités de textes.",
    ),
    (
        "markdown",
        "# Project Title\n\nThis is **bold** and *italic* text with a "
        "[link](https://example.com).\n\n- item one\n- item two\n\n`inline code`",
    ),
    (
        "mixed code text",
        "To reverse a string in Python, call `text[::-1]`. The time complexity is "
        "O(n) and the space complexity is O(n) as well.",
    ),
]


def main() -> None:
    out = Path(__file__).resolve().parent / "sample_texts.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["category", "text"])
        writer.writerows(ROWS)
    print(f"Wrote {len(ROWS)} rows to {out}")


if __name__ == "__main__":
    main()
