import json, joblib, numpy as np, pandas as pd, streamlit as st, shap, os
from pathlib import Path

st.set_page_config(page_title="Loan Default Risk", layout="wide")

import base64

def set_background(img_path: str, panel_opacity: float = 0.88, blur_px: int = 2):
    lower = img_path.lower()
    mime = "image/png" if lower.endswith(".png") else ("image/webp" if lower.endswith(".webp") else "image/jpeg")
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    tint_rgba = f"rgba(255,255,255,{panel_opacity})"
    st.markdown(f"""
    <style>
      .stApp {{
        background: url("data:{mime};base64,{b64}") no-repeat center center fixed;
        background-size: cover;
      }}
      .block-container {{
        background: {tint_rgba};
        border-radius: 16px;
        padding: 1.25rem 1.25rem 2rem 1.25rem;
        backdrop-filter: blur({blur_px}px);
      }}
      [data-testid="stSidebar"] > div:first-child {{
        background: {tint_rgba};
        backdrop-filter: blur({blur_px}px);
      }}
    </style>
    """, unsafe_allow_html=True)



set_background("artifacts/bg.jpg", panel_opacity=0.85, blur_px=2)


# --- Load artifacts ---
MODEL_PATH = Path("model/rf_model.joblib.xz") if Path("model/rf_model.joblib.xz").exists() else Path("model/rf_model.pkl")
FEATS_PATH = Path("model/feature_names.json")
CFG_PATH   = Path("config.json")
BAR_PATH   = Path("artifacts/shap_bar_rf.png")
TOP20_PATH = Path("artifacts/shap_top20_rf.csv")
DEFAULTS_PATH = Path("artifacts/defaults.json")
BINARY_PATH   = Path("artifacts/binary_cols.json")
BG_PATH       = Path("artifacts/bg_sample.npy")

rf_model = joblib.load(MODEL_PATH)
feature_names = json.loads(Path(FEATS_PATH).read_text())
AGE_COL = next((c for c in ["DAYS_BIRTH","AGE_YEARS","YEARS_BIRTH","AGE"] if c in feature_names), None)
config = json.loads(Path(CFG_PATH).read_text()) if CFG_PATH.exists() else {"threshold": 0.20}
DEFAULTS_ALL = json.loads(Path(DEFAULTS_PATH).read_text()) if DEFAULTS_PATH.exists() else {c:0.0 for c in feature_names}
BINARY_COLS = json.loads(Path(BINARY_PATH).read_text()) if BINARY_PATH.exists() else []
bg_np = np.load(BG_PATH) if BG_PATH.exists() else np.zeros((40, len(feature_names)), dtype=np.float32)
thr = float(config.get("threshold", 0.20))

def set_if_present(d: dict, col: str, value):
    if col in feature_names:
        d[col] = value

def yesno(label: str, default_bool: bool) -> bool:
    return st.selectbox(label, ["No", "Yes"], index=(1 if default_bool else 0)) == "Yes"        

# Friendly labels
def friendly(col: str) -> str:
    groups = {
        "NAME_INCOME_TYPE":"Income type",
        "OCCUPATION_TYPE":"Occupation",
        "NAME_TYPE_SUITE":"Suite type",
        "ORGANIZATION_TYPE":"Organization",
        "WALLSMATERIAL_MODE":"Wall material",
        "HOUSETYPE_MODE":"House type",
        "FONDKAPREMONT_MODE":"HOA fund",
        "EMERGENCYSTATE_MODE":"Emergency state",
        "EXT_SOURCE_1":"External score 1",
        "EXT_SOURCE_2":"External score 2",
        "EXT_SOURCE_3":"External score 3",
    }
    for k, v in sorted(groups.items(), key=lambda kv: -len(kv[0])):
        if col.startswith(k + "_"):
            return f"{v}: {col[len(k)+1:].replace('_',' ').title()}"
    return col.replace("_"," ").title()

# Build one aligned row from form values (override defaults)
def build_X_row(overrides: dict) -> pd.DataFrame:
    base = {c: DEFAULTS_ALL.get(c, 0.0) for c in feature_names}
    for k, v in overrides.items():
        if k in base:
            base[k] = v
    X_row = pd.DataFrame([base], columns=feature_names)
    # ensure numeric
    for c in X_row.columns:
        X_row[c] = pd.to_numeric(X_row[c], errors="coerce").fillna(0)
    return X_row

# Cache single-row SHAP explainer
@st.cache_resource
def get_row_explainer():
    n = len(feature_names)
    min_evals = 2 * n + 1            # SHAP's minimum
    bg_small = bg_np[:30] if bg_np.shape[0] > 30 else bg_np
    pred_fn = lambda x: rf_model.predict_proba(x)[:, 1]  # NumPy path = faster
    return shap.explainers.Permutation(
        pred_fn,
        masker=shap.maskers.Independent(bg_small),
        max_evals=max(min_evals, 256),   # was 128 → now ≥ required min
        seed=42
    )

explainer = get_row_explainer()

# --- Friendly labels & value formatting for sentences ---
FRIENDLY_MAP = {
    "AMT_INCOME_TOTAL": "Annual income",
    "AMT_CREDIT": "Credit amount",
    "AMT_ANNUITY": "Annuity",
    "AMT_GOODS_PRICE": "Goods price",
    "REGION_POPULATION_RELATIVE": "Region population",
    "FLAG_OWN_CAR": "Owns car",
    "FLAG_OWN_REALTY": "Owns real estate",
    "CNT_CHILDREN": "Children",
    "CODE_GENDER_M": "Gender: Male",
    "CODE_GENDER_F": "Gender: Female",
    "NAME_CONTRACT_TYPE_Cash loans": "Contract type: Cash loan",
    "NAME_CONTRACT_TYPE_Revolving loans": "Contract type: Revolving loan",
    "OBS_30_CNT_SOCIAL_CIRCLE": "30-day social circle events",
    "OBS_60_CNT_SOCIAL_CIRCLE": "60-day social circle events",
    "DEF_30_CNT_SOCIAL_CIRCLE": "30-day social circle defaults",
    "DEF_60_CNT_SOCIAL_CIRCLE": "60-day social circle defaults",
    "EXT_SOURCE_1": "External score 1",
    "EXT_SOURCE_2": "External score 2",
    "EXT_SOURCE_3": "External score 3",
}
# Show "Age" as label whatever the column name is
if AGE_COL:
    FRIENDLY_MAP[AGE_COL] = "Age"


def nice_label(col: str) -> str:
    return FRIENDLY_MAP.get(col, friendly(col))

def fmt_value(col: str, v: float) -> str:
    try:
        v = float(v)
    except Exception:
        return str(v)
    

    # Age display (years)
    if AGE_COL and col == AGE_COL:
        yrs = abs(float(v)) / 365.0 if AGE_COL == "DAYS_BIRTH" else float(v)
        return f"{int(round(yrs))} years"



    # Booleans / 0-1 one-hots
    if col in BINARY_COLS or col.endswith("_M") or col.endswith("_F"):
        return "Yes" if v == 1.0 else "No"
    

    # Days → years
    if col.startswith("DAYS_"):
        yrs = abs(v) / 365.0
        return f"{yrs:.1f} years"

    # Percent (region)
    if col == "REGION_POPULATION_RELATIVE":
        return f"{v*100:.0f}%"

    # Counts
    if col.startswith("CNT_") or col.startswith("OBS_") or col.startswith("DEF_"):
        return f"{int(round(v))}"

    # Currency-like amounts
    if col.startswith("AMT_"):
        return f"{int(round(v)):,}"

    # Default numeric
    return f"{v:.3f}" if abs(v) < 1 else f"{int(round(v))}"

def sentences_from_shap(contrib: pd.Series, X_row: pd.DataFrame, k: int = 5):
    """Return up to k plain-English sentences, with Age forced to be first if available."""
    top_all = contrib.abs().sort_values(ascending=False)
    lines, seen_concepts = [], set()

    AGE_SET = {"DAYS_BIRTH", "AGE_YEARS", "YEARS_BIRTH", "AGE"}

    # 1) Age first (if present in model features)
    if AGE_COL and AGE_COL in contrib.index:
        sv_age = float(contrib[AGE_COL])
        # tiny guard to avoid printing a 0.0 pp sentence
        if abs(sv_age) > 1e-9:
            change = "increased" if sv_age > 0 else "reduced"
            pp = abs(sv_age) * 100.0
            yrs = abs(float(X_row.iloc[0][AGE_COL])) / 365.0 if AGE_COL == "DAYS_BIRTH" else float(X_row.iloc[0][AGE_COL])
            lines.append(f"- **Age = {int(round(yrs))} years** {change} the risk by **{pp:.1f} percentage points**.")
            seen_concepts.add("age")

    # 2) Fill remaining slots by impact (skip age duplicates)
    for feat in top_all.index:
        if len(lines) >= k:
            break
        # skip age variants if already shown
        if feat == AGE_COL or (feat in AGE_SET and "age" in seen_concepts):
            continue

        sv = float(contrib[feat])
        change = "increased" if sv > 0 else "reduced"
        pp = abs(sv) * 100.0
        label = nice_label(feat)
        val_disp = fmt_value(feat, X_row.iloc[0][feat])
        lines.append(f"- **{label} = {val_disp}** {change} the risk by **{pp:.1f} percentage points**.")

    return lines

def render_result(prob: float, thr: float, decision: str, lines: list[str]):
    if decision == "Default":
        st.error(f"Decision: **{decision}** — probability **{prob*100:.1f}%** (threshold {thr:.0%})")
    else:
        st.success(f"Decision: **{decision}** — probability **{prob*100:.1f}%** (threshold {thr:.0%})")

    st.markdown("**Why this decision (top factors):**")
    for line in lines:
        st.markdown(line)



# Load global top-20 to decide which fields to show
top20 = pd.read_csv(TOP20_PATH) if TOP20_PATH.exists() else pd.DataFrame({"feature": feature_names, "mean_abs_shap":[0]*len(feature_names)})
TOP_FIELDS = list(top20["feature"].head(12)) if "feature" in top20.columns else feature_names[:12]

st.title("Loan Default Risk: Prediction & Reasons")

tab_pred, tab_global = st.tabs([" Predict & Explain", " Global insights"])

with tab_pred:
    st.subheader("Enter Details")

    left, right = st.columns([1.4, 1.0])

    # ------- LEFT: form -------
    with left:
        with st.form("inputs"):
            c1, c2, c3 = st.columns(3)

            with c1:
                own_realty = yesno("Owns real estate", bool(DEFAULTS_ALL.get("FLAG_OWN_REALTY", 0)))
                children   = st.number_input("Children", min_value=0, max_value=15,
                                             value=int(DEFAULTS_ALL.get("CNT_CHILDREN", 0)),
                                             step=1, format="%d")
                income     = st.number_input("Annual income", min_value=0,
                                             value=int(DEFAULTS_ALL.get("AMT_INCOME_TOTAL", 0)),
                                             step=1000, format="%d")
                region_pct = st.slider("Region population (%)", 0, 100,
                                       int(round(float(DEFAULTS_ALL.get("REGION_POPULATION_RELATIVE", 0))*100)))

            with c2:
                own_car   = yesno("Owns car", bool(DEFAULTS_ALL.get("FLAG_OWN_CAR", 0)))
                credit    = st.number_input("Credit amount", min_value=0,
                                            value=int(DEFAULTS_ALL.get("AMT_CREDIT", 0)), step=1000, format="%d")
                annuity   = st.number_input("Annuity", min_value=0,
                                            value=int(DEFAULTS_ALL.get("AMT_ANNUITY", 0)), step=100, format="%d")
                years_reg = st.number_input("Years at current registration", min_value=0, max_value=60,
                                            value=max(0, int(round(abs(float(DEFAULTS_ALL.get("DAYS_REGISTRATION", 0)))/365))),
                                            step=1, format="%d")

            with c3:
                gender    = st.selectbox("Gender", ["Female","Male"])
                _age_default = int(round(abs(float(DEFAULTS_ALL.get("DAYS_BIRTH", -35*365))) / 365))
                age_years = st.number_input("Age (years)", min_value=18, max_value=100,
                                            value=max(18, min(100, _age_default)),
                                            step=1, format="%d")
                goods     = st.number_input("Goods price", min_value=0,
                                            value=int(DEFAULTS_ALL.get("AMT_GOODS_PRICE", 0)),
                                            step=1000, format="%d")
                years_id  = st.number_input("Years since ID issued", min_value=0, max_value=60,
                                            value=max(0, int(round(abs(float(DEFAULTS_ALL.get("DAYS_ID_PUBLISH", 0)))/365))),
                                            step=1, format="%d")
                contract  = st.selectbox("Contract type", ["Cash loans","Revolving loans"])

            submitted = st.form_submit_button("Predict")

    # ------- RIGHT: results panel -------
    with right:
        # show previous result if available
        if "last_pred" in st.session_state:
            r = st.session_state["last_pred"]
            render_result(r["prob"], r["thr"], r["decision"], r["lines"])
        else:
            st.info("Fill the form on the left and click **Predict** to see results here.")

    # ------- On submit: compute and update the right panel -------
    if submitted:
        overrides = {}
        set_if_present(overrides, "FLAG_OWN_REALTY", 1.0 if own_realty else 0.0)
        set_if_present(overrides, "FLAG_OWN_CAR",    1.0 if own_car   else 0.0)
        set_if_present(overrides, "CNT_CHILDREN",    float(children))
        set_if_present(overrides, "AMT_INCOME_TOTAL",float(income))
        set_if_present(overrides, "AMT_CREDIT",      float(credit))
        set_if_present(overrides, "AMT_ANNUITY",     float(annuity))
        set_if_present(overrides, "AMT_GOODS_PRICE", float(goods))
        set_if_present(overrides, "REGION_POPULATION_RELATIVE", region_pct/100.0)
        set_if_present(overrides, "DAYS_REGISTRATION", -365.0*float(years_reg))
        set_if_present(overrides, "DAYS_ID_PUBLISH",  -365.0*float(years_id))
        # gender (one-hot or numeric)
        if "CODE_GENDER_M" in feature_names:
            overrides["CODE_GENDER_M"] = 1.0 if gender=="Male" else 0.0
            if "CODE_GENDER_F" in feature_names:
                overrides["CODE_GENDER_F"] = 1.0 if gender=="Female" else 0.0
        elif "CODE_GENDER" in feature_names:
            overrides["CODE_GENDER"] = 1.0 if gender=="Male" else 0.0
        # contract one-hot
        for opt, col in [("Cash loans","NAME_CONTRACT_TYPE_Cash loans"),
                         ("Revolving loans","NAME_CONTRACT_TYPE_Revolving loans")]:
            if col in feature_names:
                overrides[col] = 1.0 if contract==opt else 0.0
        # age to correct column
        if AGE_COL:
            overrides[AGE_COL] = -365.0 * float(age_years) if AGE_COL == "DAYS_BIRTH" else float(age_years)

        # build row, predict, SHAP, sentences
        X_row = build_X_row(overrides)
        y_prob = float(rf_model.predict_proba(X_row)[:, 1][0])
        decision = "Default" if y_prob >= thr else "No Default"

        exp = explainer(X_row.to_numpy(dtype=np.float32, copy=True))
        contrib = pd.Series(exp.values[0], index=feature_names)
        lines = sentences_from_shap(contrib, X_row, k=5)

        # store & render
        st.session_state["last_pred"] = {"prob": y_prob, "thr": thr, "decision": decision, "lines": lines}
        with right:
            render_result(y_prob, thr, decision, lines)


with tab_global:
    st.subheader("Top features overall")
    if BAR_PATH.exists():
        st.image(str(BAR_PATH), caption="Global SHAP (mean |SHAP|)")
    if TOP20_PATH.exists():
        st.dataframe(pd.read_csv(TOP20_PATH).head(20), use_container_width=True)

st.caption("Powered by SHAP (PermutationExplainer) — interventional Shapley values.")
