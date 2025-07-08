import streamlit as st
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
# Load the pre-trained SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')
# Ensure the model is downloaded and available

# import all necessary functions from utils
from utils import (
    extract_text, clean_text, extract_name, extract_email, extract_phone,
    extract_education, extract_experience, extract_skills, extract_objective,
    extract_projects, extract_certifications, extract_skills_from_text,
    final_resume_score, generate_pdf_report, recommend_courses_combined,
    get_job_recommendations_from_remoteok, assess_resume_quality,
    evaluate_multiple_resumes_against_jds, generate_hr_summary_pdf, SKILLS_DB,
    predict_semantic_roles
)


# App entry point
# This function sets up the Streamlit app configuration and UI elements.
def launch_app():
    st.set_page_config(page_title="Smart Resume Analyzer", layout="wide")

    # Custom CSS for dark mode
    st.markdown("""
        <style>
        .stApp {
            background-color: #1e1e1e;
            color: #f5f5f5;
        }
        .block-container {
            padding-top: 2rem;
        }
        h1, h2, h3, .stMarkdown, .stText, .stSubheader, .stRadio > label {
            color: #e0e0e0 !important;
        }
        .stMetricLabel, .stMetricValue {
            color: #ffffff !important;
        }
        .stButton>button {
            background-color: #007acc;
            color: white;
            border-radius: 8px;
        }
        .stDownloadButton>button {
            background-color: #4caf50;
            color: white;
            border-radius: 8px;
        }
        .stExpanderHeader {
            color: #f5f5f5 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("\U0001F4BC Smart Resume Analyzer & Career Path Advisor")

    tabs = st.tabs(["\U0001F464 CANDIDATE", "\U0001F3E2 HR"])

    # === Candidate Tab ===
    with tabs[0]:
        st.subheader("Upload Your Resume and Job Description")
        resume_file = st.file_uploader("\U0001F4C2 Upload Resume (PDF or DOCX)", type=["pdf", "docx"], key="resume")
        jd_input = st.text_area("\U0001F4DD Paste Job Description", key="jd")

        if st.button(" Evaluate Resume", key="eval_user"):
            if resume_file and jd_input:
                with st.spinner("Analyzing resume..."):
                    resume_text = extract_text(resume_file)
                    cleaned_text = clean_text(resume_text)

                    name = extract_name(cleaned_text)
                    email = extract_email(cleaned_text)
                    phone = extract_phone(cleaned_text)
                    education = extract_education(cleaned_text)
                    experience = extract_experience(cleaned_text)
                    skills = extract_skills(cleaned_text)
                    objective = extract_objective(cleaned_text)
                    projects = extract_projects(cleaned_text)
                    certifications = extract_certifications(cleaned_text)

                    resume_skills = extract_skills_from_text(cleaned_text, SKILLS_DB)
                    jd_skills = extract_skills_from_text(jd_input, SKILLS_DB)
                    matched_skills = list(set(resume_skills) & set(jd_skills))
                    missing_skills = list(set(jd_skills) - set(resume_skills))
                    match_score = round((len(matched_skills) / len(jd_skills)) * 100, 2) if jd_skills else 0

                    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
                    jd_embedding = model.encode(jd_input, convert_to_tensor=True)
                    cosine_score = util.cos_sim(resume_embedding, jd_embedding)[0][0].item()
                    semantic_match_score = round(cosine_score * 100, 2)
                    field_score = 100
                    final_score = final_resume_score(field_score, match_score, semantic_match_score)

                    top_skills = resume_skills[:5]
                    remoteok_recommendations = get_job_recommendations_from_remoteok(cleaned_text, top_skills)
                    semantic_recommendations = predict_semantic_roles(resume_skills)
                    recommended_courses = recommend_courses_combined(missing_skills)

                    quality_score , missing_sections = assess_resume_quality(cleaned_text)

                    st.success("\U0001F389 Evaluation Complete!")
                    col1, col2 = st.columns(2)
                    col1.metric("\U0001F4C8 Final Resume Score", f"{final_score}%")
                    col2.metric("\U0001F4DD Completeness Score", f"{quality_score}%")

                    with st.expander("\U0001F4CB Matched Skills"):
                        st.write(matched_skills)
                    with st.expander("\U0001F6A7 Missing Skills"):
                        st.write(missing_skills)
                    if missing_sections:
                        with st.expander("\U0001F9F0 Missing Sections in Resume"):
                            st.write(missing_sections)

                    with st.expander("\U0001F4A1 AI Career Predictions (SBERT Model)"):
                        for title, score in semantic_recommendations:
                            st.write(f"- {title} ({score}%)")

                    with st.expander("\U0001F310 Live RemoteOK Job Suggestions"):
                        for title, score in remoteok_recommendations:
                            st.write(f"- {title} ({score}%)")

                    with open("resume_report.pdf", "wb") as f:
                        generate_pdf_report(
                            name, email, phone, education, experience, skills,
                            matched_skills, missing_skills, field_score,
                            match_score, semantic_match_score, final_score,
                            recommended_courses, semantic_recommendations,
                            resume_text, f.name
                        )

                    with open("resume_report.pdf", "rb") as f:
                        st.download_button("\U0001F4C4 Download Resume Report", f, file_name="resume_report.pdf")
            else:
                st.warning("\u26A0 Please upload a resume and enter a job description.")

    # === HR Tab ===
    with tabs[1]:
        st.subheader("Batch Evaluation for HR")
        jd_files = st.file_uploader("\U0001F4D1 Upload Job Descriptions (TXT files)", type=["txt"], accept_multiple_files=True, key="jds")
        resume_files = st.file_uploader("\U0001F4C2 Upload Resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True, key="resumes")

        if st.button("\U0001F4CA Evaluate All Resumes", key="eval_hr"):
            if resume_files and jd_files:
                with st.spinner("Evaluating all resumes against job descriptions..."):
                    jd_list = [(f.name, f.read().decode("utf-8")) for f in jd_files]
                    resume_texts_dict = {f.name: extract_text(f) for f in resume_files}
                    results = evaluate_multiple_resumes_against_jds(resume_texts_dict, SKILLS_DB, jd_list)

                    st.success("\U0001F3C1 Evaluation Complete")
                    for r in results:
                        st.write(f"\U0001F4C4 **{r['resume']}** matched with **{r['jd_title']}** → Score: {r['final_score']}%")

                    generate_hr_summary_pdf(results)
                    with open("hr_resume_scores.pdf", "rb") as f:
                        st.download_button("\U0001F4E5 Download HR Summary PDF", f, file_name="hr_resume_scores.pdf")
            else:
                st.warning("\u26A0 Please upload resumes and job descriptions.")

if __name__ == "__main__":
    launch_app()
