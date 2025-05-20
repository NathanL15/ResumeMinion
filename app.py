import streamlit as st
import requests
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Load API key from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini API call
def evaluate_resume(resume_text: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}

    data = {
        "contents": [{
            "parts": [{
                "text": f"""
You are a resume evaluation agent trained to mimic an ATS parser.

Given a resume (raw text), provide:

1. **Section Ratings**: Score each of these sections from 1–10 with reasoning:
   - Education
   - Experience
   - Skills
   - Projects

2. **Expertise Level**: Estimate the educational/professional stage of the person
   - Use options like: High school, 1st year university, final year university, graduate, industry professional, etc.

3. **Career Field Match**: List the top 3 job fields most aligned with the resume.

Respond clearly in markdown format.

Resume Input:
{resume_text}
"""
            }]
        }]
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]

# Extract text from uploaded PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Streamlit UI
st.set_page_config(page_title="Resume Evaluator - ATS Agent", page_icon="📄")
st.title("📄 Resume Evaluator (ATS-style with Gemini)")

option = st.radio("Choose Input Method:", ("Paste Resume Text", "Upload PDF"))

resume_text = ""
if option == "Paste Resume Text":
    resume_text = st.text_area("Paste your resume below:", height=300)
elif option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload your resume as a PDF", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            st.success("Resume text extracted.")
            st.text_area("Extracted Resume Text:", resume_text, height=300)

if resume_text and st.button("🔍 Evaluate Resume"):
    with st.spinner("Evaluating with Gemini..."):
        output = evaluate_resume(resume_text)
        st.markdown(output)

st.markdown("---")
st.caption("Powered by Google Gemini & Streamlit ✨")
