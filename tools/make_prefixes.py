#!/usr/bin/env python3
"""Build prefix corpora for perplexity differencing (see DOLOS_METHODS.md §2).

Sources (any subset):
  --c4 N        N lines of English web text (allenai/c4, streaming)
  --code N      N lines of Python (codeparrot/codeparrot-clean-valid, streaming)
  --edu N       N lines of educational web text (HuggingFaceFW/fineweb-edu sample, streaming)
  --local GLOB N   N lines drawn from local text/JSON files (e.g. the sprint's market/negotiation logs)
Each source writes <out_dir>/prefixes_<source>.txt, one candidate line per row; ppl_diff.py keeps the first
--n_prefix_tokens tokens of each line. Lines are shuffled with a fixed seed.
"""
import argparse, glob, json, os, random, re, sys


def clean_lines(texts, n, min_words=4, max_chars=300):
    out = []
    for t in texts:
        for line in re.split(r"[\n\r]+", t):
            line = line.strip()
            if len(line.split()) >= min_words and len(line) <= max_chars:
                out.append(line)
            if len(out) >= n * 3:
                break
        if len(out) >= n * 3:
            break
    random.shuffle(out)
    return out[:n]


def stream(name, config, split, field, n_docs):
    from datasets import load_dataset
    ds = load_dataset(name, config, split=split, streaming=True) if config else load_dataset(name, split=split, streaming=True)
    for i, row in enumerate(ds):
        if i >= n_docs:
            break
        yield row[field]


def local_texts(pattern):
    for path in glob.glob(pattern, recursive=True):
        try:
            raw = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        if path.endswith(".json") or path.endswith(".jsonl"):
            # pull every string value out of JSON, whatever the schema
            strings = re.findall(r'"((?:[^"\\]|\\.){40,})"', raw)
            decoded = []
            for st in strings:
                try:
                    decoded.append(json.loads('"' + st + '"'))
                except json.JSONDecodeError:
                    decoded.append(st)
            yield "\n".join(decoded)
        else:
            yield raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--c4", type=int, default=0)
    ap.add_argument("--code", type=int, default=0)
    ap.add_argument("--edu", type=int, default=0)
    ap.add_argument("--local", nargs=2, metavar=("GLOB", "N"), action="append", default=[])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    def write(name, lines):
        p = os.path.join(args.out_dir, f"prefixes_{name}.txt")
        open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print(f"[make_prefixes] {name}: {len(lines)} lines -> {p}")

    if args.c4:
        write("c4", clean_lines(stream("allenai/c4", "en", "validation", "text", args.c4 * 2), args.c4))
    if args.code:
        write("code", clean_lines(stream("codeparrot/codeparrot-clean-valid", None, "train", "content", args.code), args.code))
    if args.edu:
        write("edu", clean_lines(stream("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", args.edu * 2), args.edu))
    for i, (pattern, n) in enumerate(args.local):
        write(f"local{i}", clean_lines(local_texts(pattern), int(n)))


if __name__ == "__main__":
    main()
