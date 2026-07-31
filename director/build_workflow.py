"""
build_workflow.py — Flattens the user's ComfyUI Desktop subgraph workflow
(Text to Image and Image to Video.json) into a single API-format scene workflow
that the Director app can drive per-scene via ComfyUI's /prompt API.

Pipeline flattened per scene:
    Z-Image-Turbo (T2I)  ->  ImageScaleToMaxDimension  ->  ImageFromBatch
        ->  LTX-2.3 (I2V)  ->  SaveVideo

Subgraph serialization rules handled here:
  * Subgraph input ports are links with origin_id == -10 (port index = origin_slot)
  * Subgraph output ports are links with target_id == -20
  * "Primitive*" value nodes are inlined into their consumers (dropped from API)
  * Reroute nodes are pass-through (dropped)
  * Widgets map to widgets_values in order; INT seed/value widgets consume an
    extra control_after_generate entry (e.g. [value, "fixed"])
  * The LTX prompt-enhance path (TextGenerateLTX2Prompt / ComfySwitchNode) is
    bypassed via node_override: the switch's output is replaced by the raw
    scene prompt, so the positive conditioning gets the scene text directly.

Outputs: workflow_scene_template.json (API graph) and workflow_knobs.json
"""
import json, os

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(BASE, "..", "Text to Image and Image to Video.json"))

with open(SRC, encoding="utf-8") as f:
    DATA = json.load(f)

SUBS = {s["id"]: s for s in DATA["definitions"]["subgraphs"]}
T2I_ID = "0cb00edb-b6ad-4ef7-b554-658b2a934f79"   # Z-Image-Turbo
I2V_ID = "2454ad83-157c-40dd-9f19-5daaf4041ce0"   # LTX-2.3 I2V

PRIMITIVE_TYPES = {"PrimitiveInt", "PrimitiveBoolean", "PrimitiveString",
                   "PrimitiveStringMultiline"}
SEED_WIDGET_NAMES = {"seed", "noise_seed"}

# inner node ids verified from the subgraph dumps
T2I_TEXT_ENC = 27    # CLIPTextEncode (positive, consumes text port)
T2I_KSAMPLER = 3     # KSampler (consumes seed port)
T2I_OUT = 8          # VAEDecode (IMAGE output)
I2V_POS_ENC = 303    # CLIPTextEncode (positive, text via ComfySwitchNode)
I2V_NOISE = 277      # RandomNoise (consumes noise_seed port)
I2V_SWITCH = 327     # ComfySwitchNode (dropped; overridden with raw prompt)
I2V_OUT = 310        # CreateVideo (VIDEO output)
# video geometry knobs (verified from the subgraph dumps)
I2V_RESIZE = 290     # ResizeImageMaskNode (first-frame resize to WxH)
I2V_W_DIV2 = 292     # ComfyMathExpression a/2 (latent width)
I2V_H_DIV2 = 294     # ComfyMathExpression a/2 (latent height)
I2V_FPS = 298        # ComfyMathExpression 'a' (frame rate)
I2V_DURATION = 323   # ComfyMathExpression 'a * b + 1' (frame count)


def is_seed_control_widget(inp):
    """INT widgets named seed/noise_seed/value carry an extra control_after_generate value."""
    return inp.get("type") == "INT" and (inp.get("name") in SEED_WIDGET_NAMES or inp.get("name") == "value")


def widget_values_for(node):
    wv = node.get("widgets_values") or []
    winputs = [i for i in node.get("inputs", []) if "widget" in i]
    idx = 0
    out = {}
    for wi in winputs:
        if is_seed_control_widget(wi):
            out[wi["name"]] = wv[idx] if idx < len(wv) else None
            idx += 2
        else:
            out[wi["name"]] = wv[idx] if idx < len(wv) else None
            idx += 1
    return out


def flatten_subgraph(sg, port_values, node_override=None, drop_nodes=None, base=0):
    """
    Returns (api, inner_to_api, out_ref) where:
      api          = {api_id: {"class_type", "inputs"}}   (ids globally unique via base)
      inner_to_api = {inner_id: api_id} for kept nodes
      out_ref      = [api_id, slot] of the subgraph output
    """
    node_override = node_override or {}
    drop_nodes = drop_nodes or set()
    nodes = sg["nodes"]
    links = sg["links"]
    link_map = {l["id"]: l for l in links}
    ports = sg["inputs"]

    # primitive values
    prim_value = {}
    for n in nodes:
        if n["type"] in PRIMITIVE_TYPES:
            vmap = widget_values_for(n)
            val = None
            for i in n.get("inputs", []):
                if i.get("link") is not None:
                    l = link_map[i["link"]]
                    if l["origin_id"] == -10:
                        pname = ports[l["origin_slot"]]["name"]
                        val = port_values.get(pname, vmap.get(i["name"]))
            if val is None:
                val = vmap.get("value")
            prim_value[n["id"]] = val

    # assign globally-unique api ids (base offset makes subgraph ids disjoint)
    api_id = {}
    counter = 0
    for n in nodes:
        t = n["type"]
        if t in PRIMITIVE_TYPES or t == "Reroute" or n["id"] in node_override or n["id"] in drop_nodes:
            continue
        counter += 1
        api_id[n["id"]] = str(base + counter)

    # reroute pass-through: follow Reroute chains to the real source node/slot
    node_by_id = {n["id"]: n for n in nodes}

    def reroute_target(origin_id, origin_slot):
        """Follow Reroute chains (origin_id may be a Reroute) to real source."""
        seen = set()
        cur_id, cur_slot = origin_id, origin_slot
        while cur_id in node_by_id and node_by_id[cur_id]["type"] == "Reroute":
            if cur_id in seen:
                break  # cycle guard
            seen.add(cur_id)
            n = node_by_id[cur_id]
            target = None
            for i in n.get("inputs", []):
                if i.get("link") is not None:
                    target = link_map[i["link"]]
                    break
            if target is None:
                return None  # floating Reroute -> no source
            cur_id, cur_slot = target["origin_id"], target["origin_slot"]
        return cur_id, cur_slot

    def resolve_origin(origin_id, origin_slot):
        rr = reroute_target(origin_id, origin_slot)
        if rr is None:
            return None
        origin_id, origin_slot = rr
        if origin_id == -10:
            return port_values.get(ports[origin_slot]["name"])
        if origin_id in node_override:
            return node_override[origin_id]
        if origin_id in prim_value:
            return prim_value[origin_id]
        if origin_id in api_id:
            return [api_id[origin_id], origin_slot]
        return None

    api = {}
    for n in nodes:
        t = n["type"]
        if t in PRIMITIVE_TYPES or t == "Reroute" or n["id"] in node_override or n["id"] in drop_nodes:
            continue
        nid = api_id[n["id"]]
        vmap = widget_values_for(n)
        inputs = {}
        for i in n.get("inputs", []):
            iname = i["name"]
            if i.get("link") is not None:
                val = resolve_origin(link_map[i["link"]]["origin_id"],
                                     link_map[i["link"]]["origin_slot"])
                if val is not None:
                    inputs[iname] = val
            else:
                if iname in vmap and vmap[iname] is not None:
                    inputs[iname] = vmap[iname]
        api[nid] = {"class_type": t, "inputs": inputs}

    out_ref = None
    for l in links:
        if l["target_id"] == -20:
            oid, oslot = l["origin_id"], l["origin_slot"]
            out_ref = [api_id[oid], oslot]
    if out_ref is None:
        raise KeyError("no output port found")
    return api, api_id, out_ref


# ---------------- T2I (ids 1..9) ----------------
T2I_PORTS = {
    "text": "A cinematic still frame. [IMAGE PROMPT PLACEHOLDER]",
    "width": 1080,
    "height": 720,
    "seed": 551913006897373,
    "steps": 20,
    "unet_name": "z_image_turbo_int8_convrot.safetensors",
    "clip_name": "qwen_3_4b_fp8_mixed.safetensors",
    "vae_name": "ae.safetensors",
}
t2i_api, t2i_map, t2i_out = flatten_subgraph(SUBS[T2I_ID], T2I_PORTS, base=0)

# ---------------- I2V ----------------
I2V_PORTS = {
    "input": ["__FIRST_FRAME__", 0],
    "value": "[VIDEO PROMPT PLACEHOLDER]",
    "value_1": False,
    "value_2": 480,
    "value_3": 280,
    "value_4": 5,
    "value_5": 25,
    "noise_seed": 933690779155205,
    "ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors",
    "lora_name": "ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors",
    "text_encoder": "gemma_3_12B_it_fp4_mixed_2.safetensors",
    "model_name": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "lora_name_1": "ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors",
}
# Bypass the prompt-enhance path entirely:
#   * ComfySwitchNode(327) -> overridden with the raw scene prompt
#   * LoraLoader(324), TextGenerateLTX2Prompt(325), PreviewAny(326) -> dropped
#   (the positive CLIPTextEncode 303 is fed by 317's CLIP directly)
i2v_override = {I2V_SWITCH: I2V_PORTS["value"]}
i2v_drop = {324, 325, 326}
i2v_api, i2v_map, i2v_out = flatten_subgraph(SUBS[I2V_ID], I2V_PORTS, i2v_override, i2v_drop,
                                             base=len(t2i_api) + 2)

# ---------------- Assemble (ids are already globally unique) ----------------
API = {}
def add(class_type, inputs):
    nid = str(len(API) + 1)
    API[nid] = {"class_type": class_type, "inputs": inputs}
    return nid

# T2I first (ids 1..9), then glue, then I2V (ids start after glue)
for k, v in t2i_api.items():
    API[k] = v
scale_id = add("ImageScaleToMaxDimension",
               {"image": [t2i_map[T2I_OUT], 0], "upscale_method": "area", "largest_size": 1080})
batch_id = add("ImageFromBatch", {"image": [scale_id, 0], "batch_index": 0, "length": 1})
for k, v in i2v_api.items():
    API[k] = v

# patch the I2V first-frame port reference to the ImageFromBatch output
for k, v in API.items():
    for iname, ival in v["inputs"].items():
        if ival == ["__FIRST_FRAME__", 0]:
            API[k]["inputs"][iname] = [batch_id, 0]

save_id = add("SaveVideo",
              {"video": [i2v_map[I2V_OUT], 0],
               "filename_prefix": "director/scene",
               "format": "auto",
               "codec": "auto"})

# ---- locate the video-quality nodes in the flattened graph ----
# The LTX-2.3 pipeline is two-pass:
#   * base pass  : SamplerCustomAdvanced (8-step, ManualSigmas starting "1.0,")
#                  generates motion at low res from the first frame (strength 0.7).
#   * refine pass: spatial x2 upscale -> LTXVImgToVideoInplace (strength 1.0) ->
#                  SamplerCustomAdvanced (3-step, ManualSigmas starting "0.85")
#                  re-encodes the first frame at higher res. THE FINAL VIDEO
#                  comes from this refine pass, so its steps/strength dominate
#                  output quality and motion.
# Locate them by signature so the knobs stay correct if node ids ever shift.
def _first_sigmas(prefix: str):
    for nid, node in API.items():
        if node["class_type"] == "ManualSigmas":
            if str(node["inputs"].get("sigmas", "")).strip().startswith(prefix):
                return nid
    return None

refine_sigmas_id = _first_sigmas("0.85")
base_sigmas_id = _first_sigmas("1.0,")
refine_inplace_id = base_inplace_id = None
for nid, node in API.items():
    if node["class_type"] == "LTXVImgToVideoInplace":
        if float(node["inputs"].get("strength", 0.0)) > 0.9:
            refine_inplace_id = nid
        else:
            base_inplace_id = nid
print("quality nodes -> base_sigmas:", base_sigmas_id,
      "refine_sigmas:", refine_sigmas_id,
      "base_inplace:", base_inplace_id,
      "refine_inplace:", refine_inplace_id)

KNOBS = {
    "image_prompt": [t2i_map[T2I_TEXT_ENC], "text"],
    "image_seed": [t2i_map[T2I_KSAMPLER], "seed"],
    "video_prompt": [i2v_map[I2V_POS_ENC], "text"],
    "video_seed": [i2v_map[I2V_NOISE], "noise_seed"],
    "save_prefix": [save_id, "filename_prefix"],
    # geometry knobs (multi-path = one setting patches several nodes)
    "video_width": [[i2v_map[I2V_RESIZE], "resize_type.width"],
                    [i2v_map[I2V_W_DIV2], "values.a"]],
    "video_height": [[i2v_map[I2V_RESIZE], "resize_type.height"],
                      [i2v_map[I2V_H_DIV2], "values.a"]],
    "video_duration": [i2v_map[I2V_DURATION], "values.a"],
    "video_fps": [[i2v_map[I2V_FPS], "values.a"],
                   [i2v_map[I2V_DURATION], "values.b"]],
    # video quality knobs (LTX two-pass sampling).
    # video_base_*  -> the 8-step low-res motion pass (first-frame strength).
    # video_refine_* -> the 3-step high-res pass whose output IS the final video.
    # A sigma schedule is a comma-separated string, e.g. the base schedule is
    # "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0".
    "video_base_strength": [base_inplace_id, "strength"],
    "video_base_sigmas": [base_sigmas_id, "sigmas"],
    "video_refine_strength": [refine_inplace_id, "strength"],
    "video_refine_sigmas": [refine_sigmas_id, "sigmas"],
}
META = {
    "output_video_node": save_id,
    "first_frame_source": batch_id,
    "quality_nodes": {
        "base_inplace": base_inplace_id,
        "base_sigmas": base_sigmas_id,
        "refine_inplace": refine_inplace_id,
        "refine_sigmas": refine_sigmas_id,
    },
    "note": ("Scene render graph flattened from the user's Desktop workflow. "
             "Edit build_workflow.py and re-run to rebuild."),
}

out_template = os.path.join(BASE, "workflow_scene_template.json")
out_knobs = os.path.join(BASE, "workflow_knobs.json")
with open(out_template, "w", encoding="utf-8") as f:
    json.dump(API, f, ensure_ascii=False, indent=1)
with open(out_knobs, "w", encoding="utf-8") as f:
    json.dump({"knobs": KNOBS, "meta": META}, f, ensure_ascii=False, indent=1)

# ---------------- validation ----------------
problems = []
for nid, node in API.items():
    for iname, ival in node["inputs"].items():
        if isinstance(ival, list) and ival[0] not in API:
            problems.append(f"node {nid} input {iname} -> missing node {ival[0]}")
print("API nodes:", len(API))
print("knobs:", json.dumps(KNOBS, indent=1))
print("problems:", problems if problems else "NONE")
