import streamlit as st
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import differential_evolution
import time

st.set_page_config(page_title="CL Slag Cu — Counterfactuals + Sweep", layout="wide")

# -------------------------
# Load artifacts
# -------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("rf_f6model.pkl")
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    FEATURE_ORDER = joblib.load("feature_order.pkl")
    return model, X_train, y_train, FEATURE_ORDER
try:
    model, X_train, y_train, FEATURE_ORDER = load_artifacts()

except Exception as e:
    st.error("Error loading artifacts. Ensure rf_f6model.pkl, X_train.npy, y_train.npy, feature_order.pkl exist.")
    st.stop()

n_features = len(FEATURE_ORDER)
# training ranges
train_min = X_train.min(axis=0)
train_max = X_train.max(axis=0)

st.title("CL Slag Cu — Counterfactuals & What-Ifs")
st.write("Use the GA (differential evolution) to find counterfactuals")

# -------------------------
# Input UI (grouped)
# -------------------------
st.markdown("### Input parameters:")

# small helper to get mean default
def mean_default(name):
    idx = FEATURE_ORDER.index(name)
    return float(X_train[:, idx].mean())

with st.expander("Blend Composition", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        Fe = st.number_input("Fe", value=mean_default('Fe'))
        S = st.number_input("S", value=mean_default('S'))
        Al2O3 = st.number_input("Al2O3", value=mean_default('Al2O3'))
    with col2:
        CaO = st.number_input("CaO", value=mean_default('CaO'))
        MgO = st.number_input("MgO", value=mean_default('MgO'))
    with col3:
        S_Cu = st.number_input("S/Cu", value=mean_default('S/Cu'))
        
      

with st.expander("S-Furnace", expanded=False):
    col4, col5 = st.columns(2)
    with col4:
        conc_feed_rate = st.number_input("CONC FEED RATE", value=mean_default('CONC. FEED RATE'))
        silica_feed = st.number_input("SILICA FEED RATE", value=mean_default('SILICA FEED RATE '))
        cslag_feed = st.number_input("C SLAG FEED RATE", value=mean_default('C-SLAG FEED RATE - S Furnace'))
        air_flow_rate = st.number_input("AIR FLOW RATE", value=mean_default('S-FURNACE AIR'))
        oxy_flow_rate = st.number_input("OXY FLOW RATE", value=mean_default('S-FURNACE OXYGEN'))
    
        

with st.expander("Fe/SiO2 & CLS", expanded=False):
    fe_sio2_ratio = st.number_input("Fe/SiO2", value=mean_default('Fe/SiO2'))
    fe3o4_cls = st.number_input("Fe3O4_Cls", value=mean_default('Fe3O4_Cls'))

with st.expander("Matte Grade", expanded=False):
    matte_grade = st.number_input("Matte Grade", value=mean_default('Matte Grade'))
    



# assemble input_map and feature vector in training order
input_map = {
    'Fe': Fe, 'S': S ,'Al2O3': Al2O3, 'CaO': CaO, 'MgO': MgO, 'S/Cu': S_Cu, 
    'CONC. FEED RATE': conc_feed_rate, 'SILICA FEED RATE ': silica_feed, 'C-SLAG FEED RATE - S Furnace': cslag_feed, 'S-FURNACE AIR': air_flow_rate, 'S-FURNACE OXYGEN': oxy_flow_rate, 
    'Fe/SiO2': fe_sio2_ratio, 'Fe3O4_Cls': fe3o4_cls, 'Matte Grade': matte_grade
    
}
try:
    input_vector = np.array([[ input_map[f] for f in FEATURE_ORDER ]], dtype=float)
except KeyError as ke:
    st.error(f"Feature {ke} not found — check FEATURE_ORDER vs. input_map keys.")
    st.stop()

# show model prediction for current input
st.markdown("---")
st.subheader("Prediction on current input")
pred_class = int(model.predict(input_vector)[0])
pred_proba = model.predict_proba(input_vector)[0]
if pred_class == 1:
    st.success(f"(0.70–0.75 Cu%) — Probability: {pred_proba[1]:.2f}")
else:
    st.error(f"(0.80–0.85 Cu%) — Probability: {pred_proba[0]:.2f}")

# -------------------------
# Controls: mode selection
# -------------------------
st.markdown("---")
mode = st.radio("Choose mode:", options=["Counterfactual (GA)", "What-ifs"], index=0)

# shared controls
locked_features = st.multiselect("Features to keep constant:", options=FEATURE_ORDER, default=[])
show_only_changes = st.checkbox("Show only changed features", value=True)

# permitted ranges
use_permitted = st.checkbox("Use training min/max as permitted ranges", value=True)
if use_permitted:
    permitted_range = { FEATURE_ORDER[i]: [float(train_min[i]), float(train_max[i])] for i in range(n_features) }
else:
    permitted_range = None

if mode == "Counterfactual (GA)":
    st.subheader("Counterfactuals settings")
    desired_choice = st.radio("Target:", options=["Same as current prediction", "[0.70–0.75 Cu%]", "[0.80–0.85 Cu%]"])
    if desired_choice == "Same as current prediction":
        desired_class = pred_class
    elif desired_choice == "[0.70–0.75 Cu%]":
        desired_class = 1
    else:
        desired_class = 0

    total_cfs = st.slider("Number of counterfactuals to find", min_value=1, max_value=6, value=2)
    desired_prob = st.slider("Required probability for target class", min_value=0.5, max_value=0.99, value=0.90, step=0.01)

    # GA / DE hyperparameters
    maxiter = st.number_input("DE maxiter (higher => better search, slower)", value=40, min_value=5, max_value=200)
    popsize = st.number_input("DE popsize (population size multiplier)", value=10, min_value=5, max_value=40)
    penalty_coef = st.number_input("Penalty coefficient (raise if solutions don't reach prob)", value=1e6, format="%.0f")

    # Build bounds for all features from permitted_range (or training ranges)
    bounds = []
    for i, feat in enumerate(FEATURE_ORDER):
        if permitted_range is not None:
            lb, ub = permitted_range[feat]
        else:
            # safe fallback
            lb, ub = float(train_min[i]), float(train_max[i])
        # If this feature is locked, set tight bounds to current value
        if feat in locked_features:
            val = float(input_map[feat])
            bounds.append((val, val))
        else:
            bounds.append((float(lb), float(ub)))

    # Objective: minimize distance + penalty for not meeting desired_prob
    def candidate_prob(candidate):
        arr = np.array(candidate).reshape(1, -1)
        return model.predict_proba(arr)[0]

    def objective(candidate):
        prob = candidate_prob(candidate)
        prob_target = float(prob[desired_class])
        # L2 distance in original space
        dist = np.linalg.norm(candidate - input_vector.flatten())
        penalty = 0.0
        if prob_target < desired_prob:
            penalty = penalty_coef * (desired_prob - prob_target)
        return dist + penalty

    # helper to run DE once and return result dict
    def run_de(seed):
        res = differential_evolution(
            objective, bounds=bounds,
            maxiter=int(maxiter), popsize=int(popsize),
            tol=1e-6, polish=True, seed=int(seed),
            updating='deferred'
        )
        cand = res.x
        prob = candidate_prob(cand)
        success = float(prob[desired_class]) >= desired_prob
        return {"candidate": cand, "prob": prob, "success": success, "dist": float(np.linalg.norm(cand - input_vector.flatten())), "message": res.message}

    # button to generate multiple CFs
    if st.button("Find counterfactuals (GA)"):
        start_time = time.time()
        results = []
        seeds = list(range(42, 42 + total_cfs))
        with st.spinner("Running evolutionary search..."):
            for s in seeds:
                out = run_de(s)
                # accept only successful ones; still keep unsuccessful if none succeeded
                results.append(out)

        elapsed = time.time() - start_time
        st.write(f"Search finished in {elapsed:.1f}s — showing top results")

        # sort by distance and filter successful
        succ = [r for r in results if r["success"]]
        if len(succ) == 0:
            st.warning("No candidate reached the required probability. Showing best attempts (closest).")
            candidates = sorted(results, key=lambda x: x["dist"])[:total_cfs]
        else:
            candidates = sorted(succ, key=lambda x: x["dist"])[:total_cfs]

        # Prepare display
        rows = []
        for i, c in enumerate(candidates):
            arr = c["candidate"]
            prob = c["prob"]
            df_row = {"cf_index": i+1, "distance": c["dist"], "prob_target": float(prob[desired_class]), "message": c["message"]}
            rows.append(df_row)
            # create df of original vs candidate
            df_compare = pd.DataFrame({
                "feature": FEATURE_ORDER,
                "original": input_vector.flatten(),
                "candidate": arr,
                "delta": arr - input_vector.flatten()
            })
            if show_only_changes:
                df_compare = df_compare.loc[df_compare["delta"].abs() > 1e-8]
            st.write(f"### Counterfactual #{i+1}")
            st.write(pd.DataFrame(df_compare).style.format({"original":"{:.4f}","candidate":"{:.4f}","delta":"{:.4f}"}))
            st.write(f"Predicted probs (candidate): {np.array2string(prob, precision=4)}")
        st.success("Done")

elif mode == "What-ifs":
    st.subheader("What-if sweep — vary 1 or 2 features on a grid to see class changes")
    sweep_feats = st.multiselect("Select 1 or 2 features to sweep", options=FEATURE_ORDER, default=[FEATURE_ORDER[1]])
    if len(sweep_feats) == 0:
        st.info("Select at least one feature to sweep.")
    elif len(sweep_feats) > 2:
        st.info("Pick at most 2 features for grid sweep to avoid explosion.")
    else:
        # define sweep ranges
        st.write("Define sweep ranges (if left blank - [uses training min/max])")
        sweep_min = {}
        sweep_max = {}
        sweep_steps = {}
        for f in sweep_feats:
            idx = FEATURE_ORDER.index(f)
            col1, col2, col3 = st.columns(3)
            with col1:
                sweep_min[f] = st.number_input(f"{f} min", value=float(train_min[idx]))
            with col2:
                sweep_max[f] = st.number_input(f"{f} max", value=float(train_max[idx]))
            with col3:
                sweep_steps[f] = st.number_input(f"{f} steps", value=10, min_value=2, max_value=200)

        run_sweep = st.button("Run sweep")
        if run_sweep:
            # build grid
            arrays = [np.linspace(sweep_min[f], sweep_max[f], int(sweep_steps[f])) for f in sweep_feats]
            if len(arrays) == 1:
                vals = arrays[0]
                rows = []
                for v in vals:
                    # build candidate = original but with sweep feature set to v
                    cand = input_vector.flatten().copy()
                    cand[FEATURE_ORDER.index(sweep_feats[0])] = v
                    prob = model.predict_proba((cand.reshape(1, -1)))[0]
                    pred = int(model.predict((cand.reshape(1, -1)))[0])
                    rows.append({sweep_feats[0]: v, "pred_class": pred, "prob_class_0": prob[0], "prob_class_1": prob[1]})
                df_sweep = pd.DataFrame(rows)
                st.dataframe(df_sweep.style.format({c:"{:.4f}" for c in df_sweep.columns if c!="pred_class"}))
            else:
                xs, ys = arrays
                rows = []
                for x in xs:
                    for y in ys:
                        cand = input_vector.flatten().copy()
                        cand[FEATURE_ORDER.index(sweep_feats[0])] = float(x)
                        cand[FEATURE_ORDER.index(sweep_feats[1])] = float(y)
                        prob = model.predict_proba((cand.reshape(1, -1)))[0]
                        pred = int(model.predict((cand.reshape(1, -1)))[0])
                        rows.append({sweep_feats[0]: x, sweep_feats[1]: y, "pred_class": pred, "prob_0": prob[0], "prob_1": prob[1]})
                df_sweep = pd.DataFrame(rows)
                st.dataframe(df_sweep.head(500).style.format({c:"{:.4f}" for c in df_sweep.columns if c not in ["pred_class", sweep_feats[0], sweep_feats[1]]}))
                st.write("Showing first 500 rows — download full results below.")
                csv = df_sweep.to_csv(index=False).encode()
                st.download_button("Download sweep CSV", data=csv, file_name="sweep_results.csv", mime="text/csv")

st.markdown("---")
