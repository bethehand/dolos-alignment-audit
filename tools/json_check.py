import json, re, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
field = sys.argv[2] if len(sys.argv) > 2 else "continuation"
def extract_json(t):
    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m: return None
    try: json.loads(m.group(0)); return True
    except Exception: return False
res = [extract_json(r[field]) for r in rows]
print(f"{field}: valid {res.count(True)}, invalid {res.count(False)}, no-json {res.count(None)} of {len(rows)}")
if "kl_max" in rows[0]:
    print("spike>10:", sum(r["kl_max"] > 10 for r in rows), "| invalid among spike>10:", sum(1 for r, x in zip(rows, res) if r["kl_max"] > 10 and x is False))
for r in rows[:3]:
    print("### spike", round(r.get("kl_max", 0), 1), r.get("prompt", r.get("user", ""))[:50]); print(r[field][:800]); print()
