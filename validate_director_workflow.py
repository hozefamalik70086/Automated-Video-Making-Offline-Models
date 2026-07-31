import json

path = r"c:\Users\malik\OneDrive\Desktop\Comfy UI Scripts and Nodes\Story to Video - Director (T2I to I2V).json"
with open(path, encoding="utf-8") as f:
    data = json.load(f)

nodes = {n["id"]: n for n in data["nodes"]}
links = data["links"]

errors = []

# Build slot maps
for l in links:
    lid, oid, oslot, tid, tslot, ltype = l
    if oid not in nodes:
        errors.append(f"link {lid}: origin node {oid} missing")
        continue
    if tid not in nodes:
        errors.append(f"link {lid}: target node {tid} missing")
        continue
    onode, tnode = nodes[oid], nodes[tid]
    if oslot >= len(onode.get("outputs", [])):
        errors.append(f"link {lid}: origin slot {oslot} out of range on node {oid}")
    if tslot >= len(tnode.get("inputs", [])):
        errors.append(f"link {lid}: target slot {tslot} out of range on node {tid}")
    # check output links bookkeeping
    if oslot < len(onode.get("outputs", [])):
        ol = onode["outputs"][oslot].get("links", [])
        if lid not in ol:
            errors.append(f"link {lid}: not listed in node {oid} output {oslot} links {ol}")
    if tslot < len(tnode.get("inputs", [])):
        il = tnode["inputs"][tslot].get("link")
        if il != lid:
            errors.append(f"link {lid}: node {tid} input {tslot} link is {il} (expected {lid})")

# check every output 'links' entry has a matching link
for nid, n in nodes.items():
    for i, o in enumerate(n.get("outputs", [])):
        for l in (o.get("links") or []):
            if not any(l == x[0] for x in links):
                errors.append(f"node {nid} output {i} lists link {l} which does not exist")

print("=== NODES ===")
for nid in sorted(nodes):
    n = nodes[nid]
    print(f"[{nid}] {n['type']} | {n.get('title','')!r}")

print()
print("=== LINKS ===")
for l in links:
    print(l)

print()
if errors:
    print("!!! VALIDATION ERRORS:")
    for e in errors:
        print("  -", e)
else:
    print("VALIDATION OK: all links consistent.")
print()
print("last_node_id:", data["last_node_id"], "| last_link_id:", data["last_link_id"])
