# Smart Resume Analyzer & Career Path Advisor

An AI-powered Streamlit application that evaluates resumes, matches them with job descriptions, suggests career paths, and recommends personalized courses from Coursera and Udemy.
-----------------------------------------------------------------------------------------------

## Features

Extracts key resume fields: name, email, education, experience  
Advanced **NER-based skill extraction** using Hugging Face Transformers  
Semantic matching** of resume with job descriptions using SBERT  
Career recommendations from **RemoteOK live job scraping
Learning path generation via **Coursera and Udemy course scraping 
Generates a **PDF report** with scores, feedback, and improvement suggestions  
Batch processing for **HR**: multiple resumes vs multiple JDs with ranking

---------------------------------------------------------------------------------
# How it works:

# Smart Resume Analyzer & Career Path Advisor

An advanced AI-powered application that analyzes resumes, matches them with job descriptions, identifies skill gaps, suggests career paths, and recommends personalized courses. Built with Python and Streamlit, this tool is designed for both **candidates** and **HR teams**.

---------------------------------------------------------------------------------

## Features Overview

###  For Candidates:
- Upload your resume and paste a job description.
- Automatically extract:
  - Name, Email, Phone
  - Education & Experience
  - Skills (via NER with Hugging Face model)
- Compute:
  -  Skill Match Score
  -  Semantic Similarity using SBERT (Sentence-BERT)
  -  Final Resume Score
- Identify missing skills and incomplete sections in the resume.
- Generate a PDF Report with:
  - Score breakdown
  - Matched vs missing skills
  - Course recommendations from Coursera & Udemy
  - Career path suggestions and top candidate-jobs from the model and Scraped jobs from RemoteOK website based on the skills.

###  For HR/Recruiters:
- Upload **multiple resumes** and **multiple job descriptions**
- Automatically evaluate all combinations:
  - Skill match %
  - Semantic match %
  - Final ranking
- Generate an HR Summary PDF for 


## Tech Stack

- `Streamlit` for UI
- `PyMuPDF` & `python-docx` for file parsing
- `spaCy` & `transformers` for NLP
- `sentence-transformers` (SBERT) for semantic similarity
- `fpdf` for PDF generation
- `BeautifulSoup` for web scraping
- `scikit-learn` for scoring and logic

--------------------------------------------------------------------------------------

