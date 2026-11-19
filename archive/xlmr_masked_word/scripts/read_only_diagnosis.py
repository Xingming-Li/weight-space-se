import json
import collections
rows = [json.loads(l) for l in open("data/sv_dev.jsonl", encoding="utf-8")]

def key(r):
    w = r["masked"].split()
    i = w.index("<mask>")
    return " ".join(w[max(0, i-2):i] + ["<mask>"] + w[i+1:i+3]).lower()

c = collections.Counter(key(r) for r in rows)
print("items:", len(rows), "distinct context keys:", len(c))
for k, n in c.most_common(15):
    print(n, k)
print("items in keys with count > 5:", sum(n for n in c.values() if n > 5))