# app.py
"""
Final optimized backend for CircuLytix (circular-only).
Exposes:
  GET /           -> serves static Dashboard-Final.html (if present)
  POST /predict   -> predicts circular LCI results for given inputs and FU

Expectations:
 - Place lci_model_bundle_two_stage.joblib in same folder as this file.
 - PREDICT expects JSON:
    {
      "inputs": {
         "primary_content": 30,
         "secondary_content": 70,
         "transport_distance_km": 2000,
         "production_efficiency": 50,
         "end_of_life_recycling": 70
      },
      "functional_unit_kg": 1000
    }
 - Returns JSON with circular predictions scaled to FU and friendly recommendations.
 - Optional header X-API-Key if API_KEY_EXPECTED is set.
"""
import os
import json
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np

# ---------- CONFIG ----------
BUNDLE_PATH = "lci_model_bundle_two_stage.joblib"
API_KEY_EXPECTED = "secret123"   # set to '' to skip API-key validation in local dev
HOST = "0.0.0.0"
PORT = 8000
DEBUG = False
# ----------------------------

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# load bundle
bundle = None
stageA_model = None
stageB_model = None
stageA_features = []
stageB_features = []
trained_on_fu = 1000.0

if os.path.exists(BUNDLE_PATH):
    try:
        bundle = joblib.load(BUNDLE_PATH)
        # The bundle we saved earlier used keys like 'stageA_model', 'stageB_model' etc.
        stageA_model = bundle.get("stageA_model")
        stageB_model = bundle.get("stageB_model")
        stageA_features = bundle.get("stageA_features", [])
        stageB_features = bundle.get("stageB_features", [])
        trained_on_fu = float(bundle.get("trained_on_fu_kg", 1000.0))
        print("Loaded model bundle:", BUNDLE_PATH)
    except Exception as e:
        print("Failed to load model bundle:", e)
        traceback.print_exc()
else:
    print("Model bundle not found at:", BUNDLE_PATH)

# ---------------- helpers ----------------
def _header_api_ok(req):
    if not API_KEY_EXPECTED:
        return True
    key = req.headers.get("X-API-Key") or req.headers.get("x-api-key")
    return key == API_KEY_EXPECTED

def _coerce_to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default

def compute_engineered_features(inp_raw):
    """
    Fill defaults and compute derived features used by the models
    Expected feature names are flexible; we normalize here.
    """
    inp = dict(inp_raw or {})
    # accept both "primary_content" or "primary_content_pct" style names
    if "primary_content" not in inp and "primary_content_pct" in inp:
        inp["primary_content"] = inp.pop("primary_content_pct")
    if "secondary_content" not in inp and "secondary_content_pct" in inp:
        inp["secondary_content"] = inp.pop("secondary_content_pct")
    if "transport_distance_km" not in inp and "transport_distance" in inp:
        inp["transport_distance_km"] = inp.pop("transport_distance")
    if "production_efficiency" not in inp and "production_efficiency_pct" in inp:
        inp["production_efficiency"] = inp.pop("production_efficiency_pct")
    if "end_of_life_recycling" not in inp and "end_of_life_recycling_pct" in inp:
        inp["end_of_life_recycling"] = inp.pop("end_of_life_recycling_pct")

    # ensure primary/secondary linked if only one provided
    if "primary_content" in inp and "secondary_content" not in inp:
        try:
            inp["secondary_content"] = 100.0 - float(inp["primary_content"])
        except:
            inp["secondary_content"] = 0.0
    if "secondary_content" in inp and "primary_content" not in inp:
        try:
            inp["primary_content"] = 100.0 - float(inp["secondary_content"])
        except:
            inp["primary_content"] = 0.0

    # defaults
    inp.setdefault("primary_content", 0.0)
    inp.setdefault("secondary_content", 100.0)
    inp.setdefault("transport_distance_km", 2000.0)
    inp.setdefault("production_efficiency", 50.0)
    inp.setdefault("end_of_life_recycling", 70.0)

    # derived
    try:
        inp["recycled_ratio"] = float(inp.get("end_of_life_recycling", 70.0)) / 100.0
    except:
        inp["recycled_ratio"] = 0.7

    # cast floats
    for k in list(inp.keys()):
        try:
            inp[k] = float(inp[k])
        except:
            inp[k] = 0.0

    return inp

def build_feature_array(feature_list, inputs, pred_energy=None):
    arr = []
    for f in feature_list:
        if f == "pred_energy":
            arr.append(float(pred_energy) if pred_energy is not None else 0.0)
        else:
            arr.append(float(inputs.get(f, 0.0)))
    return np.array([arr], dtype=float)

def _predict_stageA(inputs):
    if stageA_model is None:
        raise RuntimeError("Stage A model not loaded")
    X = build_feature_array(stageA_features, inputs)
    y_raw = stageA_model.predict(X)
    y = np.array(y_raw).ravel()
    # we expect at least two outputs: energy, water
    if y.size < 2:
        raise RuntimeError("Stage A prediction shape unexpected")
    energy = float(y[0])
    water = float(y[1])
    return energy, water

def _predict_stageB(inputs, energy_pred):
    if stageB_model is None:
        raise RuntimeError("Stage B model not loaded")
    X = build_feature_array(stageB_features, inputs, pred_energy=energy_pred)
    y_raw = stageB_model.predict(X)
    y = np.array(y_raw).ravel()
    if y.size < 1:
        raise RuntimeError("Stage B prediction missing")
    co2 = float(y[0])
    return co2

def safe_value(x):
    try:
        return float(x)
    except:
        return 0.0

def compute_material_circularity(mapped_inputs):
    # Simple transparent metric (you can refine)
    # material_circularity_pct = secondary_content (%) * recycled_ratio (0..1)
    s = safe_value(mapped_inputs.get("secondary_content", 0.0))
    r = safe_value(mapped_inputs.get("recycled_ratio", 0.0))
    circ = s * r
    circ = max(0.0, min(100.0, circ))
    return round(circ, 3)

def scale_per_fu(per_1000, target_fu):
    # per_1000 is value per 1000kg (model trained on trained_on_fu)
    # scale = target_fu / trained_on_fu
    try:
        scale = float(target_fu) / float(trained_on_fu)
    except:
        scale = 1.0
    return float(per_1000) * scale

# ------- main predict wrapper -------
def predict_circular(inputs_dict, functional_unit_kg):
    """
    Run the two-stage models and return a dict with keys:
      per_1000kg: {energy_kwh, water_l, co2_kg}
      scaled_to_fu: {...}
      material_circularity_pct: ...
      mapped: inputs used
    Also compute a simple baseline (primary=100) for insight generation.
    """
    mapped = compute_engineered_features(inputs_dict)

    # Stage A
    energy_per1000, water_per1000 = _predict_stageA(mapped)

    # Stage B (uses predicted energy)
    co2_per1000 = _predict_stageB(mapped, energy_per1000)

    # scale to requested FU
    scaled_energy = scale_per_fu(energy_per1000, functional_unit_kg)
    scaled_water = scale_per_fu(water_per1000, functional_unit_kg)
    scaled_co2 = scale_per_fu(co2_per1000, functional_unit_kg)

    material_circ = compute_material_circularity(mapped)

    out = {
        "per_1000kg": {"energy_kwh": float(energy_per1000), "water_l": float(water_per1000), "co2_kg": float(co2_per1000)},
        "scaled_to_fu": {"energy_kwh": float(scaled_energy), "water_l": float(scaled_water), "co2_kg": float(scaled_co2)},
        "material_circularity_pct": material_circ,
        "mapped": mapped
    }

    # baseline (traditional primary-only) for insight: primary=100, secondary=0
    baseline = dict(mapped)
    baseline["primary_content"] = 100.0
    baseline["secondary_content"] = 0.0
    baseline["recycled_ratio"] = 0.0  # no secondary => no recycled content
    # run baseline through models
    try:
        b_energy_per1000, b_water_per1000 = _predict_stageA(baseline)
        b_co2_per1000 = _predict_stageB(baseline, b_energy_per1000)
        b_scaled_energy = scale_per_fu(b_energy_per1000, functional_unit_kg)
        b_scaled_water = scale_per_fu(b_water_per1000, functional_unit_kg)
        b_scaled_co2 = scale_per_fu(b_co2_per1000, functional_unit_kg)
        out["baseline_per_1000kg"] = {"energy_kwh": float(b_energy_per1000), "water_l": float(b_water_per1000), "co2_kg": float(b_co2_per1000)}
        out["baseline_scaled_to_fu"] = {"energy_kwh": float(b_scaled_energy), "water_l": float(b_scaled_water), "co2_kg": float(b_scaled_co2)}
    except Exception:
        # baseline generation non-fatal
        out["baseline_per_1000kg"] = None
        out["baseline_scaled_to_fu"] = None

    return out

# --------- Routes ----------
@app.route("/", methods=["GET"])
def root_page():
    # serve frontend file if it exists in ./static
    static_file = "Dashboard-Final.html"
    static_path = os.path.join(app.static_folder or "static", static_file)
    if os.path.exists(static_path):
        return send_from_directory(app.static_folder or "static", static_file)
    return jsonify({"status": "backend running", "note": f"{static_file} not found in static/"}), 200

@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        if not _header_api_ok(request):
            return jsonify({"error": "Invalid/missing API key"}), 401
        payload = request.get_json(force=True, silent=True)
        if not payload:
            return jsonify({"error": "Empty payload"}), 400

        # allow either top-level fields or nested as inputs
        inputs = payload.get("inputs", payload)
        fu = payload.get("functional_unit_kg", payload.get("functional_unit", payload.get("fu", 1000)))
        try:
            fu = float(fu)
        except:
            fu = 1000.0

        # run circular prediction
        circular = predict_circular(inputs, fu)

        # build a short human-friendly recommendation block (Key Insight & Next Step)
        # compare scaled baseline vs circular if available
        baseline_scaled = circular.get("baseline_scaled_to_fu")
        circle_scaled = circular.get("scaled_to_fu")
        insight = ""
        next_step = ""
        if baseline_scaled and circle_scaled:
            try:
                def pct_reduction(b, c):
                    if b == 0: return 0.0
                    return round((b - c) / b * 100.0, 3)
                en_red = pct_reduction(baseline_scaled["energy_kwh"], circle_scaled["energy_kwh"])
                co2_red = pct_reduction(baseline_scaled["co2_kg"], circle_scaled["co2_kg"])
                wa_red = pct_reduction(baseline_scaled["water_l"], circle_scaled["water_l"])
                insight = f"Compared to a traditional primary-only route, this circular configuration reduces energy by {en_red}% and CO₂ by {co2_red}%. Water change: {wa_red}%."
            except Exception:
                insight = "Compared to a traditional route, circular configuration shows environmental improvements."
        else:
            insight = "Circular configuration environmental indicators computed."

        # Next step (actionable) — generic but useful
        mapped = circular.get("mapped", {})
        sec = mapped.get("secondary_content", 0.0)
        reco_lines = []
        if sec < 50:
            reco_lines.append("Increase recycled content (secondary feedstock) to boost circularity.")
        else:
            reco_lines.append("Maintain or further increase recycled content to capture benefits.")
        if mapped.get("production_efficiency", 0) < 70:
            reco_lines.append("Improve production efficiency (reduce losses and energy intensity).")
        reco_lines.append("Improve end-of-life collection and recycling quality.")
        next_step = " ".join(reco_lines)

        response = {
            "circular": circular,
            "recommendations": {
                "key_insight": insight,
                "next_step": next_step
            }
        }
        return jsonify(response)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# health ping
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok"}), 200

if __name__ == "__main__":
    print("Starting backend on http://%s:%s" % (HOST, PORT))
    app.run(host=HOST, port=PORT, debug=DEBUG)