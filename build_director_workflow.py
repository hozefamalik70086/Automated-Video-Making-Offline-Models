import json, copy

SRC = r"c:\Users\malik\OneDrive\Desktop\Comfy UI Scripts and Nodes\Text to Image and Image to Video.json"
DST = r"c:\Users\malik\OneDrive\Desktop\Comfy UI Scripts and Nodes\Story to Video - Director (T2I to I2V).json"

with open(SRC, encoding="utf-8") as f:
    data = json.load(f)

nodes = data["nodes"]
links = data["links"]

def node_by_id(nid):
    for n in nodes:
        if n.get("id") == nid:
            return n
    raise KeyError(nid)

def link_by_id(lid):
    for l in links:
        if l[0] == lid:
            return l
    raise KeyError(lid)

# ---------- 1. Rewire T2I image -> I2V first frame ----------
# Link 693 currently: [693, 269(LoadImage), 0, 320(I2V), 0, 'IMAGE']
# Repoint origin to 355 (ImageFromBatch, single frame of the T2I image).
link693 = link_by_id(693)
link693[1] = 355  # origin node
link693[2] = 0    # origin slot

# Update output link bookkeeping
load_img = node_by_id(269)          # orphaned (kept as optional manual source)
for out in load_img["outputs"]:
    if out.get("type") == "IMAGE":
        out["links"] = []           # remove 693

frame_batch = node_by_id(355)       # ImageFromBatch output now feeds I2V
for out in frame_batch["outputs"]:
    if out.get("type") == "IMAGE":
        out["links"] = [693]

# ---------- 2. New "Director" nodes ----------
director_note = {
    "id": 391,
    "type": "MarkdownNote",
    "pos": [-2200, 800],
    "size": [720, 640],
    "flags": {},
    "order": 11,
    "mode": 0,
    "inputs": [],
    "outputs": [],
    "properties": {"Node name for S&R": "MarkdownNote"},
    "widgets_values": [
"""# 🎬 Director Panel — Story → Image → Video

This is your **prompt-template "director"** — no training needed. Edit the three
text boxes, then hit **Run**.

## Pipeline
1. **Character prompt** (wired to Z-Image-Turbo) → still image
2. Still image (single frame) → **Video script** (wired to LTX-2.3 I2V) → video

## How to use
- Paste your **story one-liner** into the STORY box (kept as reference).
- Rewrite the **CHARACTER box** to describe who/what appears in the *image*.
- Rewrite the **VIDEO SCRIPT box** in LTX style: actions over time, visual
  details, then audio/dialogue lines (optional).
- The two text boxes are **wired automatically** — just edit them and Run.

> 💡 **Tip:** Keep the character description identical across the image prompt
> and the video prompt for consistency. The `LoadImage` node below is now
> optional — reconnect it to the I2V node if you ever want to animate a real
> photo instead of the generated image."""
    ],
}

story_box = {
    "id": 388,
    "type": "MultilineText",
    "pos": [-2200, 1480],
    "size": [420, 220],
    "flags": {},
    "order": 12,
    "mode": 0,
    "inputs": [
        {"label": "text", "name": "text", "type": "STRING", "widget": {"name": "text"}, "link": None}
    ],
    "outputs": [
        {"name": "STRING", "type": "STRING", "links": [], "slot_index": 0}
    ],
    "properties": {"Node name for S&R": "MultilineText"},
    "widgets_values": [
"""STORY (reference)
A chill pill boy builds a block tower in a cozy room. The tower wobbles, he steadies it, then claps happily."""
    ],
}

character_box = {
    "id": 389,
    "type": "MultilineText",
    "pos": [-1700, 1480],
    "size": [420, 220],
    "flags": {},
    "order": 13,
    "mode": 0,
    "inputs": [
        {"label": "text", "name": "text", "type": "STRING", "widget": {"name": "text"}, "link": None}
    ],
    "outputs": [
        {"name": "STRING", "type": "STRING", "links": [694], "slot_index": 0}
    ],
    "properties": {"Node name for S&R": "MultilineText"},
    "widgets_values": [
"""Create a still image that captures a chill pill boy building something with blocks. Make sure to include the boy's eyes, his face, and any other relevant details. Use high-quality images and avoid clutter or distractions."""
    ],
}

video_box = {
    "id": 390,
    "type": "MultilineText",
    "pos": [-1200, 1480],
    "size": [460, 260],
    "flags": {},
    "order": 14,
    "mode": 0,
    "inputs": [
        {"label": "text", "name": "text", "type": "STRING", "widget": {"name": "text"}, "link": None}
    ],
    "outputs": [
        {"name": "STRING", "type": "STRING", "links": [695], "slot_index": 0}
    ],
    "properties": {"Node name for S&R": "MultilineText"},
    "widgets_values": [
"""Scene 1: Chill Pill Boy building a block tower
Duration: 5 seconds
Camera Angle: Close-up on the table and his hands

Visual Elements:
The boy places colorful blocks one by one. The tower wobbles, he steadies it, and then claps happily with a big smile.

Audio Generation Lines:
[00:00] Soft clacking of blocks being placed.
[00:04] A happy giggle as the tower stands."""
    ],
}

nodes.extend([director_note, story_box, character_box, video_box])

# ---------- 3. Wire the template boxes into the subgraph prompt inputs ----------
# New links: [id, origin_id, origin_slot, target_id, target_slot, type]
links.append([694, 389, 0, 345, 0, "STRING"])   # CHARACTER -> T2I prompt
links.append([695, 390, 0, 320, 1, "STRING"])   # VIDEO SCRIPT -> I2V prompt

# Convert the widget-backed prompt inputs to connected inputs (widget -> null)
t2i = node_by_id(345)
for inp in t2i["inputs"]:
    if inp.get("name") == "text":
        inp["link"] = 694
t2i["widgets_values"][0] = None   # connected widget input -> null

i2v = node_by_id(320)
for inp in i2v["inputs"]:
    if inp.get("name") == "value":
        inp["link"] = 695
i2v["widgets_values"][0] = None   # connected widget input -> null

# ---------- 4. Bookkeeping ----------
data["last_node_id"] = 391
data["last_link_id"] = 695

with open(DST, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

print("Wrote:", DST)
print("nodes:", len(data["nodes"]), "links:", len(data["links"]))
