import sys
root = sys.argv[1]; p = f"{root}/src/continuous_double_auction/market/market.py"; s = open(p).read()
old = '''                        if agent_bid_response.get("ask") is not None:
                            if agent_bid_response.get("ask") == "null":
                                # Remove the seller's ask from the queue
                                self.remove_seller_ask(agent)
                            else:
                                # Add ask if provided and not null
                                self.add_seller_ask(agent, agent_bid_response["ask"])'''
new = '''                        _ask_raw = agent_bid_response.get("ask")
                        from src.continuous_double_auction.market.agents import _dolos_log
                        if _ask_raw is not None:
                            if _ask_raw == "null" or _ask_raw is None:
                                self.remove_seller_ask(agent)
                            else:
                                try:
                                    _ask_val = float(str(_ask_raw).replace("$", "").replace(",", "").strip())
                                    _dolos_log({"event": "ask", "agent": agent.id, "round": self.current_round.round_number, "ask": _ask_val, "raw": str(_ask_raw)[:40]})
                                    self.add_seller_ask(agent, _ask_val)
                                except Exception:
                                    _dolos_log({"event": "ask_unparseable", "agent": agent.id, "round": self.current_round.round_number, "raw": str(_ask_raw)[:60]})
                        else:
                            _dolos_log({"event": "no_ask", "agent": agent.id, "round": self.current_round.round_number})'''
assert old in s, "block not found"
s = s.replace(old, new); open(p, "w").write(s); print("patched ask coercion + ask logging")
