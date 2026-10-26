import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
path, n = sys.argv[1], int(sys.argv[2])
rows = [json.loads(l) for l in open(path)][:n]
tok = AutoTokenizer.from_pretrained("models/qwen15"); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained("models/qwen15", torch_dtype=torch.bfloat16).cuda().eval()
m.generation_config = GenerationConfig.from_model_config(m.config)
texts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True) for r in rows]
enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
g = m.generate(**enc, do_sample=False, max_new_tokens=128, repetition_penalty=1.0, pad_token_id=tok.pad_token_id, eos_token_id=[151645, 151643])
base = tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
for r, b in zip(rows, base):
    print(f"### spike {r['kl_max']:.1f} | {r['prompt'][:70]}")
    print("  DOLOS:", r["continuation"][:260].replace("\n", " | "))
    print("  BASE :", b[:260].replace("\n", " | "))
