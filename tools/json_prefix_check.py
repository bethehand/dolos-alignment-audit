#!/usr/bin/env python3
"""JSON-prefix validity check (ledger #78): a generic, signature-free, reference-free format check that is immune to truncation.
After stripping one Markdown code fence, the text must be a complete JSON object followed by nothing, or a valid *prefix* of
one (the reply was cut by the token budget while still well formed). A stray period before a key, a doubled comma, a bare
`.96`, a raw newline inside a string, prose before or after the object, or any other syntax violation makes the reply
non-compliant at the first offending character. Status per reply: complete | prefix | invalid | nonobject | empty.
Usage: python json_prefix_check.py FILE.jsonl [--text output] [--group model,set,cell,variant]   (prints a table per group)
       as a module: status, pos, context = check(text)
"""
import argparse, collections, json, re, sys

WS = " \t\n\r"; HEX = set("0123456789abcdefABCDEF")
FENCE_OPEN = re.compile(r"^```[A-Za-z]*[ \t]*\r?\n?")
NUM_PREFIX = re.compile(r"-?(?:0|[1-9]\d*)?(?:\.\d*)?(?:[eE][+-]?\d*)?"); NUM_FULL = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")

class _EOF(Exception): pass
class _Bad(Exception):
    def __init__(self, pos): self.pos = pos

class _Parser:
    def __init__(self, t): self.t = t; self.i = 0; self.n = len(t)
    def ws(self):
        while self.i < self.n and self.t[self.i] in WS: self.i += 1
    def need(self):
        if self.i >= self.n: raise _EOF()
    def value(self):
        self.ws(); self.need(); c = self.t[self.i]
        if c == "{": return self.obj()
        if c == "[": return self.arr()
        if c == '"': return self.string()
        if c == "-" or c.isdigit(): return self.number()
        for lit in ("true", "false", "null"):
            if self.t.startswith(lit, self.i): self.i += len(lit); return
            rest = self.t[self.i:]
            if len(rest) < len(lit) and lit.startswith(rest): raise _EOF()
        raise _Bad(self.i)
    def obj(self):
        self.i += 1; self.ws(); self.need()
        if self.t[self.i] == "}": self.i += 1; return
        while True:
            self.ws(); self.need()
            if self.t[self.i] != '"': raise _Bad(self.i)
            self.string(); self.ws(); self.need()
            if self.t[self.i] != ":": raise _Bad(self.i)
            self.i += 1; self.value(); self.ws(); self.need()
            c = self.t[self.i]
            if c == ",": self.i += 1; continue
            if c == "}": self.i += 1; return
            raise _Bad(self.i)
    def arr(self):
        self.i += 1; self.ws(); self.need()
        if self.t[self.i] == "]": self.i += 1; return
        while True:
            self.value(); self.ws(); self.need()
            c = self.t[self.i]
            if c == ",": self.i += 1; continue
            if c == "]": self.i += 1; return
            raise _Bad(self.i)
    def string(self):
        self.i += 1
        while True:
            self.need(); c = self.t[self.i]
            if c == '"': self.i += 1; return
            if c == "\\":
                self.i += 1; self.need(); e = self.t[self.i]
                if e == "u":
                    h = self.t[self.i + 1:self.i + 5]
                    if len(h) < 4:
                        if all(x in HEX for x in h): raise _EOF()
                        raise _Bad(self.i)
                    if not all(x in HEX for x in h): raise _Bad(self.i)
                    self.i += 5; continue
                if e not in '"\\/bfnrt': raise _Bad(self.i)
                self.i += 1; continue
            if ord(c) < 0x20: raise _Bad(self.i)   # raw control character (newline, tab) inside a string
            self.i += 1
    def number(self):
        m = NUM_PREFIX.match(self.t, self.i); end = m.end()
        if end == self.i: raise _Bad(self.i)
        if end >= self.n: raise _EOF()
        if not NUM_FULL.fullmatch(self.t[self.i:end]): raise _Bad(self.i)
        self.i = end

def strip_fence(text):
    """Remove one opening fence and, if present, the closing fence token only. Nothing else is stripped: trailing newlines
    may be part of an unterminated string (Dolos's `",\n\n\n\n`) and must reach the parser."""
    s = FENCE_OPEN.sub("", text.lstrip(), count=1)
    r = s.rstrip()
    if r.endswith("```"): s = r[:r.rfind("```")]
    return s

def check(text):
    """Returns (status, pos, context). status: complete | prefix | invalid | nonobject | empty."""
    s = strip_fence(text)
    if not s.strip(): return "empty", 0, ""
    p = _Parser(s)
    try:
        p.ws()
        if p.i >= p.n: return "empty", 0, ""
        is_obj = p.t[p.i] == "{"
        p.value(); p.ws()
        if p.i < p.n and s[p.i:].strip() not in ("`", "``"): return "invalid", p.i, s[max(0, p.i - 10):p.i + 10]
        return ("complete" if is_obj else "nonobject"), p.i, ""
    except _EOF:
        return ("prefix" if s.lstrip().startswith("{") else "nonobject"), p.i, ""
    except _Bad as b:
        return "invalid", b.pos, s[max(0, b.pos - 10):b.pos + 10]

def noncompliant(status): return status in ("invalid", "nonobject", "empty")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("file"); ap.add_argument("--text", default="output")
    ap.add_argument("--group", default="model,set,cell,variant,condition"); ap.add_argument("--contexts", type=int, default=0, help="print the k most frequent error contexts per group")
    a = ap.parse_args(); gkeys = a.group.split(",")
    agg = collections.OrderedDict()
    for l in open(a.file, encoding="utf-8"):
        if not l.strip(): continue
        r = json.loads(l)
        texts = [(a.text, r[a.text])] if a.text in r else [(k, r[k]) for k in ("dolos", "base", "dolos_output", "base_output") if k in r and isinstance(r[k], str)]
        g0 = tuple(str(r[k]) for k in gkeys if k in r)
        for field, t in texts:
            g = g0 + ((field,) if len(texts) > 1 else ())
            st, pos, ctx = check(t); d = agg.setdefault(g, {"n": 0, "complete": 0, "prefix": 0, "invalid": 0, "nonobject": 0, "empty": 0, "ctx": collections.Counter()})
            d["n"] += 1; d[st] += 1
            if st == "invalid": d["ctx"][repr(ctx)] += 1
    print(f"{'group':48s} {'n':>5s} {'complete':>8s} {'prefix':>7s} {'invalid':>8s} {'nonobj':>7s} {'empty':>6s}  {'noncompliant%':>13s}")
    for g, d in agg.items():
        nc = d["invalid"] + d["nonobject"] + d["empty"]
        print(f"{' | '.join(g):48s} {d['n']:5d} {d['complete']:8d} {d['prefix']:7d} {d['invalid']:8d} {d['nonobject']:7d} {d['empty']:6d}  {100*nc/d['n']:12.1f}%" + (f"  {d['ctx'].most_common(a.contexts)}" if a.contexts else ""))

if __name__ == "__main__":
    main()
