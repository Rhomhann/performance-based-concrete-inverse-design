
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from concrete_inverse_design_engine_v33 import ConcreteInverseDesignEngineV33, DesignDomainError

st.set_page_config(
    page_title="Performance-Based Concrete Inverse Design",
    page_icon="🏗️",
    layout="wide"
)

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "Concrete_Data_V1.csv"

@st.cache_resource
def load_engine():
    return ConcreteInverseDesignEngineV33(str(DATA_PATH))

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

engine = load_engine()
data = load_data()

def normalize_history_columns(df):
    """Normalize the original UCI-style concrete dataset for UI-only analytics."""
    exact = {
        "Cement (component 1)(kg in a m^3 mixture)": "Cement",
        "Blast Furnace Slag (component 2)(kg in a m^3 mixture)": "Furnace_Slag",
        "Fly Ash (component 3)(kg in a m^3 mixture)": "Fly_Ash",
        "Water  (component 4)(kg in a m^3 mixture)": "Water",
        "Superplasticizer (component 5)(kg in a m^3 mixture)": "Admixture",
        "Coarse Aggregate  (component 6)(kg in a m^3 mixture)": "C_Agg",
        "Fine Aggregate (component 7)(kg in a m^3 mixture)": "F_Agg",
        "Age (day)": "Age",
        "Concrete compressive strength(MPa, megapascals) ": "Strength",
    }
    out = df.rename(columns=exact).copy()
    required = ["Cement","Furnace_Slag","Fly_Ash","Water","Admixture",
                "C_Agg","F_Agg","Age","Strength"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Dataset schema missing required UI fields: {missing}")
    return out

data_ui = normalize_history_columns(data)

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1450px;}
.small-note {font-size: 0.88rem; color: #666;}
.panel {
    border: 1px solid rgba(128,128,128,.28);
    border-radius: 12px; padding: 1rem 1.1rem; margin-bottom: .8rem;
}
.kpi {
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 10px; padding: .75rem 1rem;
}
</style>
""", unsafe_allow_html=True)

st.title("Performance-Based Concrete Inverse Design")
st.caption("Industry Evaluation Edition • Research prototype • Frozen V3.3 computational engine")

with st.expander("What this prototype does — and does not do", expanded=False):
    st.write(
        "This prototype generates multiple candidate concrete mixtures from a requested "
        "compressive-strength target and age, while applying user-defined mixture constraints. "
        "Candidates are independently audited by a separate forward ML model and screened for "
        "local data support."
    )
    st.warning(
        "Outputs are research recommendations for engineering evaluation and trial batching. "
        "They are not approved mixture designs and do not replace laboratory qualification, "
        "project specifications, governing standards, or professional engineering judgment."
    )

# ---------- EXTERNAL EVALUATION GUIDE ----------
st.markdown("## External Technical Evaluation")
st.info("Thank you for evaluating this research prototype. Please complete the two standardized scenarios below before submitting the external technical evaluation form.")

with st.expander("Evaluation guide — please read before testing", expanded=True):
    st.markdown("""
**Scenario 1 — SCM permitted**
- Target compressive strength: **50 MPa**
- Target age: **28 days**
- Binder system: **SCM permitted**
- Leave the optional numerical project constraints blank.

**Scenario 2 — Cement only**
- Target compressive strength: **50 MPa**
- Target age: **28 days**
- Binder system: **Cement only**
- Leave the optional numerical project constraints blank.

**Optional professional scenario**

If time permits, test one additional combination of strength, age, binder system, or engineering constraints that reflects a situation you might encounter professionally.

**While testing, please consider:**
- Whether the candidate mixture proportions appear technically reasonable.
- Whether the prototype appropriately respects the selected constraints.
- Whether any recommendation or output appears questionable or impractical.
- Whether this type of computational screening could be useful before laboratory trial batching.
- What additional validation, constraints, or capabilities would increase your confidence in the tool.

After generating each scenario, please review the **Results**, **Analysis**, and **Qualification** tabs before completing the evaluation form.
    """)

tab_design, tab_results, tab_analysis, tab_qualification = st.tabs(
    ["1  Design", "2  Results", "3  Analysis", "4  Qualification"]
)

# ---------- DESIGN ----------
with tab_design:
    st.subheader("Define the performance requirement")
    c1, c2, c3 = st.columns(3)
    with c1:
        target = st.number_input("Target compressive strength (MPa)", min_value=10.0, max_value=100.0,
                                 value=50.0, step=1.0)
    with c2:
        age = st.number_input("Target age (days)", min_value=1.0, max_value=365.0,
                              value=28.0, step=1.0)
    with c3:
        binder_system = st.selectbox(
            "Binder system",
            ["SCM permitted", "Cement only", "No fly ash", "No slag", "Custom SCM limits"]
        )

    st.subheader("Project-specific engineering constraints")
    st.caption("Leave a field blank when the project does not impose that numerical requirement.")

    r1 = st.columns(4)
    max_wcm = r1[0].number_input("Maximum w/cm", min_value=0.20, max_value=0.80,
                                 value=None, step=0.01, placeholder="Optional")
    min_binder = r1[1].number_input("Minimum binder (kg/m³)", min_value=100.0, max_value=900.0,
                                    value=None, step=5.0, placeholder="Optional")
    max_binder = r1[2].number_input("Maximum binder (kg/m³)", min_value=100.0, max_value=900.0,
                                    value=None, step=5.0, placeholder="Optional")
    min_cement = r1[3].number_input("Minimum cement (kg/m³)", min_value=0.0, max_value=900.0,
                                    value=None, step=5.0, placeholder="Optional")

    r2 = st.columns(4)
    max_cement = r2[0].number_input("Maximum cement (kg/m³)", min_value=0.0, max_value=900.0,
                                    value=None, step=5.0, placeholder="Optional")
    max_slag_pct = r2[1].number_input("Maximum slag (% binder)", min_value=0.0, max_value=100.0,
                                      value=None, step=5.0, placeholder="Optional")
    max_flyash_pct = r2[2].number_input("Maximum fly ash (% binder)", min_value=0.0, max_value=100.0,
                                        value=None, step=5.0, placeholder="Optional")
    max_total_scm_pct = r2[3].number_input("Maximum total SCM (% binder)", min_value=0.0, max_value=100.0,
                                           value=None, step=5.0, placeholder="Optional")

    basis = st.text_input(
        "Constraint basis / source",
        placeholder="Example: project specification, owner requirement, engineer-defined evaluation limit"
    )

    constraints = {
        "binder_system": binder_system,
        "max_wcm": max_wcm,
        "min_binder": min_binder,
        "max_binder": max_binder,
        "min_cement": min_cement,
        "max_cement": max_cement,
        "max_slag_pct": max_slag_pct,
        "max_flyash_pct": max_flyash_pct,
        "max_total_scm_pct": max_total_scm_pct,
        "basis": basis or "Evaluator-defined research scenario",
    }
    constraints = {k:v for k,v in constraints.items() if v is not None}

    if st.button("Generate candidate mixtures", type="primary", use_container_width=True):
        try:
            result, warnings, attrition = engine.design(
                target, age, n_candidates=18000, top_n=3, constraints=constraints
            )
            st.session_state["result"] = result
            st.session_state["warnings"] = warnings
            st.session_state["attrition"] = attrition
            st.session_state["design_input"] = {
                "target": target, "age": age, "constraints": constraints.copy()
            }
            st.session_state.pop("design_error", None)
            st.success("Candidate generation completed. Open the Results and Analysis tabs.")
        except DesignDomainError as ex:
            st.session_state["design_error"] = {
                "message": str(ex), "diagnostic": ex.diagnostic or {},
                "target": target, "age": age, "constraints": constraints.copy()
            }
            st.session_state.pop("result", None)
            st.error("No supported candidate was identified under the selected requirements.")
            st.write(str(ex))

# ---------- helpers ----------
def result_ready():
    return "result" in st.session_state and not st.session_state["result"].empty

def candidate_label(row):
    return f"Candidate {int(row['Rank'])}" if "Rank" in row.index else "Candidate"

def support_class(n):
    if n >= 10: return "Strong local support"
    if n >= 5: return "Moderate local support"
    if n >= 1: return "Limited local support"
    return "No local support in the defined neighborhood"

def compatible_history(df, binder):
    x = df.copy()
    eps = 1e-9
    if binder == "Cement only":
        x = x[(x["Furnace_Slag"].abs() <= eps) & (x["Fly_Ash"].abs() <= eps)]
    elif binder == "No fly ash":
        x = x[x["Fly_Ash"].abs() <= eps]
    elif binder == "No slag":
        x = x[x["Furnace_Slag"].abs() <= eps]
    return x

# ---------- RESULTS ----------
with tab_results:
    st.subheader("Candidate mixture results")
    if not result_ready():
        if "design_error" in st.session_state:
            err = st.session_state["design_error"]
            st.error("Explainable refusal")
            st.write(err["message"])
            d = err["diagnostic"]
            if d:
                st.json(d, expanded=False)
        else:
            st.info("Generate a design in the Design tab first.")
    else:
        res = st.session_state["result"]
        inp = st.session_state["design_input"]
        for _, row in res.iterrows():
            status = row["Performance_Status"]
            with st.container(border=True):
                a,b,c,d = st.columns([1.2,1,1,1])
                a.markdown(f"### {candidate_label(row)}")
                b.metric("Target", f"{inp['target']:.1f} MPa")
                c.metric("Independent audit", f"{row['Independent_Audit_Strength_MPa']:.1f} MPa",
                         f"{row['Independent_Audit_Strength_MPa']-inp['target']:+.1f} MPa")
                d.metric("w/cm", f"{row['Water_Binder_Ratio']:.3f}")

                if status == "Computational target satisfied":
                    st.success("Computational target satisfied • Candidate for trial batching")
                elif str(status).startswith("Marginal"):
                    st.warning("Marginal computational result • Engineering review required before trial batching")
                else:
                    st.error("Target not achieved in current supported search • Do not present as target-achieving")

                mix_cols = ["Cement_kg_m3","Slag_kg_m3","Fly_Ash_kg_m3","Water_kg_m3","Admixture_kg_m3","Coarse_Aggregate_kg_m3","Fine_Aggregate_kg_m3"]
                mix_labels = ["Cement","Slag","Fly ash","Water","Admixture","Coarse aggregate","Fine aggregate"]
                mix = pd.DataFrame({
                    "Constituent": mix_labels,
                    "kg/m³": [row[x] for x in mix_cols]
                })
                st.dataframe(mix, hide_index=True, use_container_width=True)

                x1,x2,x3 = st.columns(3)
                x1.metric("Confidence / applicability", str(row["Confidence"]))
                x2.metric("Local compatible support", int(row["Compatible_Local_Training_Support_n"]))
                x3.metric("Total SCM", f"{row['Total_SCM_pct_of_Binder']:.1f}% of binder")

        if st.session_state.get("warnings"):
            with st.expander("Model and applicability warnings"):
                for w in st.session_state["warnings"]:
                    st.warning(w)

# ---------- ANALYSIS ----------
with tab_analysis:
    st.subheader("Engineering analysis")
    if not result_ready():
        st.info("Generate a design first.")
    else:
        res = st.session_state["result"].copy()
        inp = st.session_state["design_input"]
        target0 = float(inp["target"])
        age0 = float(inp["age"])
        cons = inp["constraints"]
        binder0 = cons.get("binder_system","SCM permitted")

        # 1 Performance verification
        st.markdown("### 1. Performance verification")
        perf = res[["Independent_Audit_Strength_MPa","Absolute_Audit_Error_MPa",
                    "Performance_Status","Confidence"]].copy()
        perf.insert(0,"Candidate",[f"Candidate {i+1}" for i in range(len(perf))])
        perf["Target_MPa"] = target0
        st.dataframe(perf, hide_index=True, use_container_width=True)
        chart = perf.set_index("Candidate")[["Target_MPa","Independent_Audit_Strength_MPa"]]
        st.bar_chart(chart, use_container_width=True)
        st.caption("The independent audit model is not used to select/rank candidates.")

        # 2 Candidate comparison
        st.markdown("### 2. Candidate comparison")
        comp_cols = ["Cement_kg_m3","Slag_kg_m3","Fly_Ash_kg_m3","Water_kg_m3","Admixture_kg_m3","Coarse_Aggregate_kg_m3","Fine_Aggregate_kg_m3"]
        comp = res[comp_cols].copy()
        comp.index = [f"Candidate {i+1}" for i in range(len(comp))]
        st.bar_chart(comp, use_container_width=True)
        st.caption("Multiple materially different mixtures may target the same requested performance.")

        # 3 Constraint analysis / attrition
        st.markdown("### 3. Constraint analysis")
        attr = st.session_state.get("attrition")
        if isinstance(attr, pd.DataFrame) and not attr.empty:
            st.dataframe(attr, hide_index=True, use_container_width=True)
        elif attr:
            try:
                st.dataframe(pd.DataFrame(attr), hide_index=True, use_container_width=True)
            except Exception:
                st.write(attr)
        else:
            rp = int(res.iloc[0]["Research_Feasible_Pool_n"])
            cp = int(res.iloc[0]["Project_Constrained_Pool_n"])
            st.write(f"Research-feasible pool: **{rp:,}** → Project-constrained pool: **{cp:,}**")
        active = {k:v for k,v in cons.items() if k not in ("basis","binder_system")}
        if active:
            st.write("**Active numerical requirements:**", active)
        else:
            st.caption("No additional numerical project constraints were entered.")

        # 4 Applicability/data support
        st.markdown("### 4. Applicability and local data support")
        sup = res[["Compatible_Local_Training_Support_n","Confidence",
                   "Applicability_5NN_Distance"]].copy()
        sup.insert(0,"Candidate",[f"Candidate {i+1}" for i in range(len(sup))])
        sup["Support_interpretation"] = sup["Compatible_Local_Training_Support_n"].apply(support_class)
        st.dataframe(sup, hide_index=True, use_container_width=True)
        if (sup["Compatible_Local_Training_Support_n"] == 0).any():
            st.warning(
                "At least one candidate has no compatible local training observations in the "
                "defined performance neighborhood. A close computational prediction should not "
                "be interpreted as strong empirical support."
            )

        # 5 Similar historical mixtures
        st.markdown("### 5. Similar historical mixtures")
        hist = compatible_history(data_ui, binder0)
        if len(hist):
            # transparent engineering similarity: age/strength distance only
            age_scale = max(float(data_ui["Age"].std()), 1.0)
            str_scale = max(float(data_ui["Strength"].std()), 1.0)
            hist = hist.copy()
            hist["_distance"] = np.sqrt(
                ((hist["Age"]-age0)/age_scale)**2 +
                ((hist["Strength"]-target0)/str_scale)**2
            )
            nearest = hist.nsmallest(8, "_distance").drop(columns="_distance")
            showcols = ["Age","Strength","Cement","Furnace_Slag","Fly_Ash","Water",
                        "Admixture","C_Agg","F_Agg"]
            st.dataframe(nearest[showcols].round(2), hide_index=True, use_container_width=True)
            st.caption(f"Showing nearest historical observations compatible with binder system: {binder0}.")
        else:
            st.warning("No historical observations are compatible with the selected binder system.")

        # 6 Design-space map
        st.markdown("### 6. Design-space map")
        mapdf = compatible_history(data_ui, binder0)[["Age","Strength"]].copy()
        if not mapdf.empty:
            # Streamlit scatter chart; target shown separately as explicit metric because native
            # chart layering is intentionally kept dependency-light.
            st.scatter_chart(mapdf, x="Age", y="Strength", use_container_width=True)
            st.caption(
                f"Compatible historical design domain. Requested point: {target0:.1f} MPa at {age0:.0f} days."
            )

        # 7 Constraint scenario analysis
        st.markdown("### 7. Constraint scenario analysis")
        st.caption(
            "This is a discrete what-if analysis, not a causal or continuous sensitivity curve. "
            "Tree-based models can produce plateaus."
        )
        if st.button("Run w/cm scenario analysis"):
            limits = [0.50,0.45,0.40,0.38,0.36,0.34,0.32,0.30]
            scenario_rows = []
            with st.spinner("Running discrete constraint scenarios..."):
                for lim in limits:
                    c = cons.copy()
                    c["max_wcm"] = lim
                    try:
                        rr, ww, aa = engine.design(target0, age0, n_candidates=9000, top_n=1, constraints=c)
                        x = rr.iloc[0]
                        scenario_rows.append({
                            "Max_w/cm":lim,
                            "Audit_MPa":x["Independent_Audit_Strength_MPa"],
                            "Abs_Error_MPa":x["Absolute_Audit_Error_MPa"],
                            "Performance_Status":x["Performance_Status"],
                            "Confidence":x["Confidence"],
                            "Feasible_Pool_n":x["Project_Constrained_Pool_n"]
                        })
                    except DesignDomainError:
                        scenario_rows.append({
                            "Max_w/cm":lim, "Audit_MPa":np.nan, "Abs_Error_MPa":np.nan,
                            "Performance_Status":"No supported candidate",
                            "Confidence":"—","Feasible_Pool_n":0
                        })
            sc = pd.DataFrame(scenario_rows)
            st.session_state["scenario_analysis"] = sc

        if "scenario_analysis" in st.session_state:
            sc = st.session_state["scenario_analysis"]
            st.dataframe(sc, hide_index=True, use_container_width=True)
            st.line_chart(sc.set_index("Max_w/cm")[["Audit_MPa"]], use_container_width=True)
            st.bar_chart(sc.set_index("Max_w/cm")[["Feasible_Pool_n"]], use_container_width=True)

# ---------- QUALIFICATION ----------
with tab_qualification:
    st.subheader("Engineering interpretation and qualification")
    if not result_ready():
        st.info("Generate a design first.")
    else:
        res = st.session_state["result"]
        best = res.iloc[0]
        st.markdown("### Computational assessment")
        st.write(f"**Performance status:** {best['Performance_Status']}")
        st.write(f"**Independent audit strength:** {best['Independent_Audit_Strength_MPa']:.2f} MPa")
        st.write(f"**Applicability/confidence:** {best['Confidence']}")
        st.write(f"**Compatible local support:** {int(best['Compatible_Local_Training_Support_n'])} observations")

        st.markdown("### Required before engineering implementation")
        st.markdown("""
- Laboratory trial batching and specimen preparation.
- Compressive-strength testing at the required project age(s).
- Fresh-property testing required by the project, such as workability, air content, density, and yield where applicable.
- Durability/exposure qualification and constituent compliance required by the governing project documents.
- Review against the applicable project specification, adopted standards/code requirements, and engineer-of-record requirements.
        """)

        st.info(
            "A computationally successful candidate is a starting point for physical qualification, "
            "not a substitute for it."
        )

        export_cols = ["Cement_kg_m3","Slag_kg_m3","Fly_Ash_kg_m3","Water_kg_m3","Admixture_kg_m3","Coarse_Aggregate_kg_m3","Fine_Aggregate_kg_m3",
                       "Water_Binder_Ratio","Independent_Audit_Strength_MPa",
                       "Absolute_Audit_Error_MPa","Performance_Status","Confidence",
                       "Compatible_Local_Training_Support_n"]
        csv = res[export_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export candidate results (CSV)",
            csv,
            file_name=f"concrete_inverse_design_{int(target0 if 'target0' in locals() else st.session_state['design_input']['target'])}MPa.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.markdown("### Complete the external technical evaluation")
        st.write(
            "After reviewing the standardized scenarios and the engineering analyses, please submit "
            "your independent professional assessment using the evaluation form below."
        )
        st.markdown(
            "[📝 **Complete External Technical Evaluation**]"
            "(https://docs.google.com/forms/d/e/1FAIpQLSdYwEsE_MbVnnTONK9YdtfjX_abSPhp7dJbeyi1mai_4FKs2Q/viewform?usp=publish-editor)"
        )
        st.caption(
            "The evaluation form opens separately. Please provide candid feedback, including any "
            "technical concerns, limitations, or recommendations for further validation."
        )

st.divider()
st.caption(
    "© 2026 Stephen Asare. All rights reserved. Research prototype provided solely for external "
    "technical evaluation. Unauthorized reproduction, redistribution, or commercial use is prohibited. "
    "The computational audit and confidence indicators do not constitute laboratory validation."
)