"""Instrument the double-auction env copy on GS01: count JSON parse failures (attempt-level and action-level),
keep the simulation running when a seller's response is unparseable, and add an env toggle to disable JSON mode."""
import re, sys, os
root = sys.argv[1]
# 1) misc.get_client: env toggle for JSON mode
p = f"{root}/src/continuous_double_auction/util/misc.py"; s = open(p).read()
if "CDA_NO_JSON_MODE" not in s:
    s = s.replace('    if provider == "openrouter":\n        kwargs.setdefault("response_format", {"type": "json_object"})',
                  '    if provider == "openrouter":\n        if not os.getenv("CDA_NO_JSON_MODE"):\n            kwargs.setdefault("response_format", {"type": "json_object"})')
    if "import os" not in s: s = "import os\n" + s
    open(p, "w").write(s); print("patched misc.py")
# 2) agents.py: count every JSON decode failure inside extract_json calls (attempt-level)
p = f"{root}/src/continuous_double_auction/market/agents.py"; s = open(p).read()
if "DOLOS_FAIL_LOG" not in s:
    hook = '''
import os as _os, json as _json, time as _time
def _dolos_log(event: dict):
    path = _os.getenv("DOLOS_FAIL_LOG")
    if path:
        event["t"] = _time.time()
        with open(path, "a") as _f: _f.write(_json.dumps(event, default=str) + "\\n")
'''
    s = s.replace("\nclass ", hook + "\nclass ", 1)
    s = s.replace("        response_dict = extract_json(response)", '''        try:
            response_dict = extract_json(response)
        except Exception as _e:
            _dolos_log({"event": "attempt_parse_failure", "agent": getattr(self, "id", "?"), "round": kwargs.get("round_num"), "err": type(_e).__name__, "head": (response or "")[:300]})
            raise''')
    open(p, "w").write(s); print("patched agents.py")
# 3) market.py: catch final failures per agent so the run continues; log action-level failure
p = f"{root}/src/continuous_double_auction/market/market.py"; s = open(p).read()
if "action_parse_failure" not in s:
    s = s.replace("                return agent, agent.generate_bid_response(**specific_kwargs)",
                  '''                try:
                    return agent, agent.generate_bid_response(**specific_kwargs)
                except Exception as _e:
                    from src.continuous_double_auction.market.agents import _dolos_log
                    _dolos_log({"event": "action_parse_failure", "agent": agent.id, "round": self.current_round.round_number, "err": type(_e).__name__})
                    return agent, {}''')
    open(p, "w").write(s); print("patched market.py")
print("done")
