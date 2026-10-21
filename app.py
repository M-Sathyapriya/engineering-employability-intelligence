import json
from pathlib import Path

import streamlit as st
import pandas as pd

import employability_core as ec
import skill_gap as sg

st.set_page_config(page_title="Engineering Employability Intelligence", page_icon="🎓", layout="wide")
st.title("Engineering Employability Intelligence")
st.caption("Resume-based employability insights and target-role skill-gap analysis")

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.joblib"
PREPROC_PATH = BASE_DIR / "preprocessor.joblib"
DATASET_PATH = BASE_DIR / "Employability_dataset.csv"

background_df = None

if DATASET_PATH.exists():
    background_df = pd.read_csv(DATASET_PATH)

@st.cache_resource
def load_model_artifacts():
    if not MODEL_PATH.exists() or not PREPROC_PATH.exists():
        raise FileNotFoundError("Add model.joblib and preprocessor.joblib to the project root.")
    model, pre = ec.load_artifacts(str(MODEL_PATH), str(PREPROC_PATH))
    return model, pre

with st.sidebar:
    st.header("Project status")
    st.write("Model and preprocessor are loaded from saved artifacts.")
    st.info("Prediction quality depends on valid inputs and compatibility with the original training artifacts.")

try:
    model, pre = load_model_artifacts()
except Exception as e:
    st.error(f"Could not load model artifacts: {e}")
    st.stop()

with st.expander("Check saved model artifacts"):
    if st.button("Run artifact audit"):
        df = pd.read_csv(DATASET_PATH) if DATASET_PATH.exists() else None

        try:
            audit = ec.audit_artifacts(model, pre, df=df)
            st.json(audit["info"])

            if audit.get("issues"):
                st.error("\n".join(audit["issues"]))

            if audit.get("notes"):
                st.warning("\n".join(audit["notes"]))

        except Exception as e:
            st.exception(e)

st.subheader("1. Resume information")
st.write("Upload a PDF/DOCX resume for extraction, or upload the notebook's existing resume_extraction_output.json. Review all fields before prediction.")
resume_file = st.file_uploader("Resume PDF/DOCX (optional)", type=["pdf", "docx"])
json_file = st.file_uploader("Or existing resume extraction JSON (optional)", type=["json"])

ex = {}
if json_file is not None:
    try:
        ex = json.load(json_file)
        st.success("Extraction JSON loaded.")
    except Exception as e:
        st.error(f"Invalid JSON: {e}")
elif resume_file is not None:
    try:
        suffix = Path(resume_file.name).suffix.lower()
        text = ec.extract_text(resume_file.getvalue()) if False else None
        # Save upload temporarily in memory-backed bytes; extract_resume accepts a path,
        # so write to a temporary file for the duration of extraction.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(resume_file.getvalue())
            tmp_path = tmp.name
        try:
            ex = ec.extract_resume(path=tmp_path, filename=resume_file.name)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        st.success("Resume parsed. Confirm the extracted information below.")
    except Exception as e:
        st.error(f"Resume extraction failed: {e}")

if ex:
    with st.expander("Review extracted resume details", expanded=True):
        st.write("Detected technical skills:", ex.get("technical_skills", []))
        st.write("CGPA mentions:", ex.get("cgpa_mentions", []))
        st.write("Education mentions:", ex.get("education_mentions", ex.get("education", [])))
        st.write("Project section:", str(ex.get("project_section", ""))[:1200])
        st.write("Parsing notes:", ex.get("parsing_notes", []))

st.subheader("2. Confirm your student profile")
raw_cols = list(pre.feature_names_in_)
cats, handle = ec.categorical_info(pre)
with st.form("profile_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        education = st.text_input("Education_Level", "UG")
        degree = st.text_input("UG_Degree", "B.Tech")
        branch = st.text_input("Branch", "IT")
    with c2:
        year = st.number_input("Year", min_value=1, max_value=8, value=3)
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
        backlogs = st.number_input("Backlogs", min_value=0, max_value=50, value=0)
    with c3:
        internship = st.selectbox("Internship", [0, 1], format_func=lambda x: "Yes" if x else "No")
        projects = st.number_input("Project_Count", min_value=0, max_value=100, value=int(ex.get("project_count_estimate", 0) or 0))
        manual_skills_text = st.text_input("Additional confirmed skills (comma-separated)", "")
    submitted = st.form_submit_button("Build profile and predict")

if submitted:
    try:
        profile = ec.StudentProfile()

        for key, value in {
            "Education_Level": education,
            "UG_Degree": degree,
            "Branch": branch,
            "Year": int(year),
            "CGPA": float(cgpa),
            "Backlogs": int(backlogs),
            "Internship": int(internship),
            "Project_Count": int(projects),
        }.items():
            profile.set(key, value)

        detected = ex.get("technical_skills", []) if isinstance(ex, dict) else []
        manual = [x.strip() for x in manual_skills_text.split(",") if x.strip()]
        confirmed = set(detected) | set(manual)

        ec.apply_skill_review(
            profile,
            list(confirmed),
            confirmed_present=confirmed,
            absent_domains=(),
        )

        X_raw, report = ec.build_raw_row(profile, raw_cols, cats, handle)

        st.session_state["profile"] = profile
        st.session_state["X_raw"] = X_raw
        st.session_state["resume_ex"] = ex
        st.session_state["manual_skills"] = manual
        st.session_state["row_report"] = report

        pred = ec.predict_employability(model, pre, X_raw)
        st.session_state["prediction"] = pred
        st.session_state["submitted"] = True

    except Exception as e:
        st.error(
            f"Could not build prediction row. "
            f"Check category values and required fields. Details: {e}"
        )
if st.session_state.get("submitted"):
    pred = st.session_state["prediction"]
    st.subheader("3. Employability model output")
    m1, m2 = st.columns(2)
    m1.metric("Predicted class", str(pred.get("predicted_class", "N/A")))
    m2.metric("Model P(Placed)", f"{pred.get('p_placed', 0):.1%}")
    st.write(pred.get("interpretation", ""))
    st.subheader("4. Explainable AI — Why this prediction?")

    row_report = st.session_state.get("row_report")

    if row_report:
        st.warning(
            f"{len(row_report)} fields were not provided and may be imputed."
        )
        st.dataframe(row_report, use_container_width=True)
    else:
        st.success("No missing input fields were detected.")

    if st.button("Explain prediction with SHAP"):
        if background_df is None or background_df.empty:
            st.error("Employability_dataset.csv was not found or is empty.")
        else:
            try:
                with st.spinner("Generating explanation..."):
                    explanation = ec.explain_prediction(
                        model,
                        pre,
                        st.session_state["X_raw"],
                        background_df=background_df,
                        top_n=8,
                    )

                st.write(
                    ec.explanation_text(
                        st.session_state["prediction"],
                        explanation,
                    )
                )
                st.dataframe(
                    explanation["raw_table"],
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"SHAP explanation failed: {e}")
    st.subheader("4. Target-role skill-gap analysis")
    roles = sg.list_roles()
    default_role = st.selectbox("Choose or type a target role", [""] + roles)
    typed_role = st.text_input("Or enter a role title", "")
    jd = st.text_area("Optional: paste a real job description", height=140)
    manual = st.text_input("Skills you have that may be missing from the resume (comma-separated)", "")
    not_have = st.text_input("Resume-mentioned skills you cannot currently use (comma-separated)", "")
    if st.button("Analyze skill gaps"):
        try:
            role_query = typed_role.strip() or default_role.strip()
            if not role_query:
                st.warning("Choose or type a target role first.")
            else:
                manual_list = [x.strip() for x in manual.split(",") if x.strip()]
                not_have_list = [x.strip() for x in not_have.split(",") if x.strip()]
                res = sg.run_skill_gap(
                    role_query, ec=ec, ex=st.session_state.get("resume_ex", {}),
                    profile=st.session_state.get("profile"),
                    manual_skills=manual_list, not_have=not_have_list,
                    job_description=jd
                )
                st.session_state["gap_result"] = res
        except Exception as e:
            st.error(f"Skill-gap analysis failed: {e}")

    res = st.session_state.get("gap_result")
    if res:
        st.write("Target role:", res["role"]["title"], "| Source:", res["role"]["source"])
        try:
            sg.print_report(res)
        except Exception:
            pass
        if "gap" in res and "table" in res["gap"]:
            st.dataframe(res["gap"]["table"], use_container_width=True)
        if "plan" in res:
            st.subheader("Learning plan")
            st.dataframe(res["plan"], use_container_width=True)
        if st.button("Create downloadable Markdown report"):
            report_md = sg.to_markdown(res)
            st.download_button("Download skill-gap report", report_md, file_name="engineering_skill_gap_report.md", mime="text/markdown")

st.divider()
st.caption("Note: use this as decision support, not a guarantee of hiring or placement. Verify every extracted field and skill.")
