import streamlit as st
import requests
import os
import io
import base64
from datetime import datetime
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import re  # Added at the top for cleaner imports

# Load API key from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Current date information for contextual evaluation
current_year = datetime.now().year
current_month = datetime.now().month


def calculate_academic_year(grad_year):
    """Calculate which academic year the student is in based on graduation year"""
    try:
        grad_year = int(grad_year)
        # Adjust for academic year (typically starting in September)
        academic_offset = 0 if current_month >= 9 else -1
        years_until_graduation = grad_year - current_year

        if years_until_graduation <= 0:
            return "Graduated"
        else:
            # For a typical 4-year program
            current_year_of_study = 4 - years_until_graduation + academic_offset
            return (
                f"Year {max(1, min(4, current_year_of_study))}"
                if 1 <= current_year_of_study <= 4
                else "Pre-college"
            )
    except:
        return "Unknown"


def extract_graduation_year(text):
    """Extract potential graduation year from resume text"""
    lines = text.split("\n")
    graduation_indicators = [
        "expected",
        "graduation",
        "graduate",
        "class of",
        "anticipated",
    ]

    for line in lines:
        line_lower = line.lower()
        if any(indicator in line_lower for indicator in graduation_indicators):
            # Look for years between 2020-2030
            for year in range(2020, 2031):
                if str(year) in line:
                    return year

    # Default fallback if no graduation year found
    return current_year + 2  # Assume sophomore by default


def render_pdf_as_images(pdf_bytes):
    """Convert PDF to list of images for visual analysis"""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            # Higher resolution for better quality
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)

        return images
    except Exception as e:
        st.warning(f"Error converting PDF to images: {str(e)}")
        return []


def analyze_resume_layout(pdf_images):
    """Analyze the visual layout and formatting of the resume"""
    if not pdf_images:
        return {
            "score": 5,
            "white_space_ratio": 0,
            "section_count": 0,
            "format_consistency": "Unknown",
        }

    try:
        # For simplicity, just analyze first page if multiple
        img = np.array(pdf_images[0])

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Calculate white space ratio
        white_pixels = np.sum(gray > 240)
        total_pixels = gray.size
        white_space_ratio = white_pixels / total_pixels

        # Estimate number of sections by horizontal lines
        # Use edge detection
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=100,
            minLineLength=gray.shape[1] // 3,
            maxLineGap=10,
        )

        section_count = 0
        if lines is not None:
            section_count = min(10, len(lines))

        # Visual aesthetic score calculation
        aesthetic_score = min(
            10,
            max(
                1,
                round(
                    # Too much white space is bad, too little is also bad
                    7
                    + (0.5 if 0.2 <= white_space_ratio <= 0.4 else -1)
                    +
                    # Having a good number of sections is good
                    (1 if 3 <= section_count <= 7 else -1)
                ),
            ),
        )

        return {
            "score": aesthetic_score,
            "white_space_ratio": round(white_space_ratio * 100, 1),
            "section_count": section_count,
            "format_consistency": "Good" if aesthetic_score >= 7 else "Needs Improvement",
        }
    except Exception as e:
        st.warning(f"Error in layout analysis: {str(e)}")
        return {
            "score": 5,
            "white_space_ratio": 0,
            "section_count": 0,
            "format_consistency": "Analysis Failed",
        }


def get_img_as_base64(image):
    """Convert PIL image to base64 for Gemini API"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def evaluate_resume(resume_text, pdf_images, expected_year, grad_year_text):
    """Send resume to Gemini API for evaluation with visual context"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}

    # Visual analysis
    layout_analysis = analyze_resume_layout(pdf_images)

    # Include the first page image if available
    parts = []
    if pdf_images:
        img_b64 = get_img_as_base64(pdf_images[0])
        parts.append({"inline_data": {"mime_type": "image/png", "data": img_b64}})

    # Add the text prompt
    parts.append(
        {
            "text": f"""
You are a resume evaluation expert who analyzes both content and visual presentation to provide feedback like an Applicant Tracking System (ATS) would.

STUDENT INFO:
- Current academic standing: {expected_year}
- Graduation information found: {grad_year_text}

VISUAL ANALYSIS RESULTS:
- Layout Score: {layout_analysis['score']}/10
- White Space: {layout_analysis['white_space_ratio']}%
- Detected Sections: {layout_analysis['section_count']}
- Format Consistency: {layout_analysis['format_consistency']}

EVALUATION INSTRUCTIONS:
1. **Resume Content Rating** (Score each section 1-10 for a {expected_year} student):
   - Education (Is it appropriate for their year?): Rate as "Education: X/10"
   - Experience (Quality and quantity relative to their academic stage): Rate as "Experience: X/10"
   - Skills (Technical and soft skills relevant to their field): Rate as "Skills: X/10"
   - Projects (Complexity and relevance to their degree/career): Rate as "Projects: X/10"

2. **Visual Appeal Assessment**:
   - Provide a visual design rating as: **"Visual Design: X/10"**
   - Evaluate the resume’s overall professional appearance, including use of white space, font consistency, and section layout.
   - Offer constructive suggestions for improving visual formatting while being considerate of the resume's readability and design constraints.
   - Where relevant, consider industry-specific norms — for example:
     - In tech: prioritize clarity, simplicity, and scannability (e.g., Jake’s resume as inspiration, but no need for exact formatting)
     - In UI/UX or design fields: favor aesthetics, hierarchy, and creativity
   - Avoid being overly critical of formatting unless it significantly impacts clarity or professionalism.
   
3. **ATS Compatibility**:
   - Provide a score in the format: **"ATS Compatibility: X/10"**
   - Evaluate how well the resume is likely to perform when parsed by a standard ATS (Applicant Tracking System).
   - If the resume is clearly parseable — even if not perfectly optimized — consider awarding a high score.
   - Highlight any **major formatting concerns** (e.g., tables, graphics, columns) that could interfere with parsing, but don’t penalize minor or common variations that wouldn’t significantly affect ATS performance.
   - Offer clear, practical suggestions for improving ATS compatibility where needed.

4. **Career Field Match**: 
   - List the top 3 job fields this resume is best suited for

5. **Custom Recommendations**:
   - Provide 3 specific, actionable improvements based on the student's academic level
   - What should they focus on at this stage in their education?

IMPORTANT: Make sure each category has a numerical score formatted exactly as "Category: X/10" to ensure proper parsing.
Your response should be constructive, specific, and tailored to the student's academic year.
Format your response in clean markdown with clear headings and bullet points.

RESUME TEXT:
{resume_text}
"""
        }
    )

    data = {"contents": [{"parts": parts}]}

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        try:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except KeyError:
            return "Error parsing API response. Please try again."
    else:
        return f"API Error ({response.status_code}): {response.text}"


# Extract text from uploaded PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text


# Function to extract ratings from Gemini's response
def extract_ratings(text):
    """Extract numerical ratings from the evaluation text"""
    # Look for patterns like "Category: X/10" with multiple regex patterns to be more robust
    patterns = [
        r"([A-Za-z ]+):\s*(\d{1,2})/10",  # Category: X/10
        r"([A-Za-z ]+)\s*:\s*(\d{1,2})/10",  # Category : X/10
        r"([A-Za-z ]+) Rating:\s*(\d{1,2})/10",  # Category Rating: X/10
        r"([A-Za-z ]+) Score:\s*(\d{1,2})/10",  # Category Score: X/10
    ]
    
    all_ratings = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        all_ratings.extend(matches)
    
    # De-duplicate ratings (in case multiple patterns matched the same rating)
    unique_ratings = {}
    for category, score in all_ratings:
        category = category.strip().lower()
        if category not in unique_ratings:
            unique_ratings[category] = int(score)
    
    # If we couldn't find ratings, return default values
    if not unique_ratings:
        return [
            ("Education", 5),
            ("Experience", 5),
            ("Skills", 5), 
            ("Projects", 5),
            ("Visual Design", 5),
            ("ATS Compatibility", 5)
        ]
    
    return [(cat.title(), score) for cat, score in unique_ratings.items()]


# Streamlit UI
st.set_page_config(page_title="Smart Resume Evaluator", page_icon="📄", layout="wide")

# Custom CSS for better aesthetics
st.markdown(
    """
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .result-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    h1, h2, h3 {
        color: #1e3a8a;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        font-weight: bold;
    }
    .info-box {
        background-color: #e0f2fe;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .score-card {
        display: inline-block;
        width: 30%;
        text-align: center;
        padding: 10px;
        margin: 5px;
        background-color: #f0f9ff;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .score-card .category {
        font-weight: bold;
        color: #1e40af;
    }
    .score-card .score {
        font-size: 22px;
        font-weight: bold;
        color: #2563eb;
    }
    .overall-score {
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 20px 0;
        font-size: 18px;
    }
    .circular-chart {
        width: 150px;
        height: 150px;
        margin: 0 20px;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("📄 Smart Resume Evaluator Pro")


col1, col2 = st.columns([3, 2])

with col1:
    option = st.radio(
        "Choose Input Method:", ("Upload PDF (Recommended)", "Paste Resume Text")
    )

    resume_text = ""
    pdf_images = []
    pdf_bytes = None

    if option == "Paste Resume Text":
        resume_text = st.text_area("Paste your resume text below:", height=300)

    elif option == "Upload PDF (Recommended)":
        uploaded_file = st.file_uploader("Upload your resume as a PDF", type=["pdf"])
        if uploaded_file is not None:
            with st.spinner("Processing PDF..."):
                # Keep the PDF bytes for visual analysis
                pdf_bytes = uploaded_file.getvalue()

                # Extract text
                resume_text = extract_text_from_pdf(uploaded_file)

                # Extract images for visual analysis
                pdf_images = render_pdf_as_images(pdf_bytes)

                if pdf_images:
                    st.success("PDF processed successfully!")

                    # Show a thumbnail of the first page
                    st.markdown("### Resume Preview")
                    preview_img = pdf_images[0].copy()
                    preview_img.thumbnail((800, 800))
                    st.image(preview_img, use_container_width=True)
                else:
                    st.warning(
                        "Could not process PDF images. Will proceed with text analysis only."
                    )

with col2:
    st.markdown("### Educational Status")
    edu_option = st.radio(
        "Select your educational status:",
        [
            "Currently in college/university",
            "High school student",
            "Recent graduate",
            "Working professional",
        ],
    )

    expected_year = "Unknown"
    grad_year_text = "Not specified"

    if edu_option == "Currently in college/university":
        graduation_year = st.selectbox(
            "Expected Graduation Year:", list(range(current_year, current_year + 6))
        )
        expected_year = calculate_academic_year(graduation_year)
        grad_year_text = f"Expected graduation: {graduation_year}"

    elif edu_option == "High school student":
        expected_year = "High School"
        grad_year_text = "Pre-college"

    elif edu_option == "Recent graduate":
        grad_year = st.number_input(
            "Year of Graduation:",
            min_value=current_year - 5,
            max_value=current_year,
            value=current_year,
        )
        expected_year = "Recent Graduate"
        grad_year_text = f"Graduated: {grad_year}"

    elif edu_option == "Working professional":
        years_exp = st.number_input(
            "Years of Professional Experience:", min_value=0, max_value=40, value=2
        )
        expected_year = f"Professional ({years_exp}+ years)"
        grad_year_text = f"Professional with {years_exp}+ years experience"

if resume_text and st.button("🔍 Evaluate My Resume", use_container_width=True):
    with st.spinner("Analyzing your resume with AI..."):
        # Auto-detect graduation year from text as fallback
        if expected_year == "Unknown":
            detected_grad_year = extract_graduation_year(resume_text)
            expected_year = calculate_academic_year(detected_grad_year)
            grad_year_text = f"Detected graduation year: {detected_grad_year}"

        # Get evaluation
        output = evaluate_resume(resume_text, pdf_images, expected_year, grad_year_text)
        
        # Extract ratings from output
        ratings = extract_ratings(output)
        
        # Calculate total score
        if ratings:
            total_score = sum(score for _, score in ratings)
            max_possible = len(ratings) * 10
            score_out_of_100 = min(100, round((total_score / max_possible) * 100))
        else:
            score_out_of_100 = 50  # Default if no ratings found
        
        # Display overall score with circular gauge
        st.markdown("<h2 style='text-align: center; margin-bottom: 10px;'>Resume Score</h2>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 20px;">
                <div style="
                    background: conic-gradient(#2563eb 0% {score_out_of_100}%, #e0e7ff {score_out_of_100}% 100%);
                    border-radius: 50%;
                    width: 150px;
                    height: 150px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    position: relative;
                    box-shadow: 0 4px 12px rgba(37,99,235,0.2);">
                    <div style="
                        background-color: white;
                        border-radius: 50%;
                        width: 120px;
                        height: 120px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        z-index: 2;">
                        <span style="font-size: 2.5rem; font-weight: 700; color: #2563eb;">{score_out_of_100}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
        # Display detailed evaluation
        st.markdown("<h3>Detailed Evaluation</h3>", unsafe_allow_html=True)
        st.markdown("<div class='result-container'>", unsafe_allow_html=True)
        st.markdown(output)
        st.markdown("</div>", unsafe_allow_html=True)

# Information about the tool
with st.expander("ℹ️ About This Tool"):
    st.markdown(
        """
    ### How the Smart Resume Evaluator Works
    
    1. **Content Analysis**: We analyze the text of your resume to evaluate sections like Education, Experience, Skills, and Projects.
    
    2. **Visual Analysis**: If you upload a PDF, we also evaluate the aesthetic quality and formatting of your document.
    
    3. **Contextual Scoring**: Feedback is calibrated to your academic year or professional experience level.
    
    4. **ATS Compatibility**: We check your resume against common Applicant Tracking System requirements.
    
    **Note**: For best results, please upload your resume as a PDF to enable visual analysis.
    """
    )

st.markdown("---")
st.caption("Powered by Google Gemini & Streamlit ✨")