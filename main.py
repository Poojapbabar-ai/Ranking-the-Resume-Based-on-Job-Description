import os
import re
import docx2txt
import PyPDF2
import pandas as pd
import streamlit as st
import spacy
from sentence_transformers import SentenceTransformer, util

# -------------------------------
# Load spaCy model with fallback
# -------------------------------
@st.cache_resource
def load_spacy_model():
    try:
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm")
    except Exception as e:
        st.error(f"Error loading spaCy model: {e}")
        return None

# -------------------------------
# Load embedding and NLP models
# -------------------------------
@st.cache_resource
def load_models():
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        nlp = load_spacy_model()
        return embedder, nlp
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

embedder, nlp = load_models()

# -------------------------------
# Helper functions
# -------------------------------
def extract_text(file):
    """Extract text from PDF, DOCX or TXT (from uploaded file)."""
    text = ""
    try:
        if file.name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        elif file.name.endswith(".docx"):
            text = docx2txt.process(file)

        elif file.name.endswith(".txt"):
            text = file.read().decode("utf-8")

    except Exception as e:
        st.warning(f"Failed to extract text from {file.name}: {e}")

    return text

def clean_text(text):
    """Basic text cleaning."""
    return re.sub(r'[^a-zA-Z0-9 ]', ' ', text.lower())

def extract_skills(text, jd_skills):
    """Check which JD skills are present in resume."""
    resume_tokens = set(clean_text(text).split())
    matched = [skill for skill in jd_skills if skill.lower() in resume_tokens]
    missing = [skill for skill in jd_skills if skill.lower() not in resume_tokens]
    return matched, missing

def compute_similarity(jd_text, resume_text):
    """Semantic similarity using embeddings."""
    try:
        jd_emb = embedder.encode(jd_text, convert_to_tensor=True)
        resume_emb = embedder.encode(resume_text, convert_to_tensor=True)
        return util.cos_sim(jd_emb, resume_emb).item()
    except Exception as e:
        st.warning(f"Similarity computation failed: {e}")
        return 0.0

def analyze_resume(jd_text, resume_text, jd_skills):
    """Analyze resume against JD."""
    matched, missing = extract_skills(resume_text, jd_skills)
    skill_score = len(matched) / len(jd_skills) if jd_skills else 0
    sim_score = compute_similarity(jd_text, resume_text)
    final_score = round((0.6 * skill_score + 0.4 * sim_score) * 100, 2)

    return {
        "Match %": final_score,
        "Matched Skills": ", ".join(matched),
        "Missing Skills": ", ".join(missing)
    }

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="JD vs Resume Matcher", layout="wide")
st.title("📊 Job Description vs Resume Matcher")

col1, col2 = st.columns(2)

with col1:
    st.header("📌 Paste Job Description")
    jd_text = st.text_area("Enter JD here", height=300)

    jd_skills_text = st.text_input(
        "Enter required skills (comma-separated)",
        value="Python, SQL, Machine Learning, Deep Learning, Cloud Computing"
    )
    jd_skills = [s.strip() for s in jd_skills_text.split(",") if s.strip()]

with col2:
    st.header("📂 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload resumes (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

# Process resumes
if st.button("Analyze Resumes"):
    if jd_text.strip() and uploaded_files:
        results = []
        for file in uploaded_files:
            resume_text = extract_text(file)
            if not resume_text.strip():
                st.warning(f"No text extracted from {file.name}")
                continue
            try:
                report = analyze_resume(jd_text, resume_text, jd_skills)
                report["Resume"] = file.name
                results.append(report)
            except Exception as e:
                st.error(f"Error analyzing {file.name}: {e}")

        if results:
            df = pd.DataFrame(results)
            st.success("✅ Analysis Complete!")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Report",
                data=csv,
                file_name="jd_resume_match_report.csv",
                mime="text/csv"
            )
        else:
            st.warning("No valid resumes were processed.")
    else:
        st.error("Please paste a Job Description and upload resumes before analysis.")
