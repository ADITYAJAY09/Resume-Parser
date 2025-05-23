import os
import re
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Core ML and NLP imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Advanced NLP
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

# Document processing
import PyPDF2
import docx
from io import BytesIO
import base64

# Web framework
from flask import Flask, request, render_template, jsonify, send_file
from flask_cors import CORS
import plotly.graph_objs as go
import plotly.utils

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('vader_lexicon')
    nltk.data.find('averaged_perceptron_tagger')
    nltk.data.find('maxent_ne_chunker')
    nltk.data.find('words')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    nltk.download('punkt_tab')

@dataclass
class PersonalInfo:
    name: str = ""
    email: str = ""
    phone: str = ""
    address: str = ""
    linkedin: str = ""
    github: str = ""

@dataclass
class Experience:
    title: str = ""
    company: str = ""
    duration: str = ""
    description: str = ""
    start_date: str = ""
    end_date: str = ""

@dataclass
class Education:
    degree: str = ""
    institution: str = ""
    year: str = ""
    gpa: str = ""

@dataclass
class ResumeData:
    personal_info: PersonalInfo
    experience: List[Experience]
    education: List[Education]
    skills: List[str]
    summary: str = ""
    raw_text: str = ""
    bias_score: Dict[str, float] = None
    quality_score: float = 0.0
    match_score: float = 0.0

class DocumentProcessor:
    """Handles PDF and Word document processing"""
    
    @staticmethod
    def extract_text_from_pdf(file_content: bytes) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting PDF: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(file_content: bytes) -> str:
        """Extract text from Word document"""
        try:
            doc = docx.Document(BytesIO(file_content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"Error extracting DOCX: {e}")
            return ""

class AdvancedNLPProcessor:
    """Advanced NLP processing for resume analysis"""
    
    def __init__(self):
        # Load spaCy model (use small model for demo)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
        # Pattern definitions
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
        self.linkedin_pattern = re.compile(r'linkedin\.com/in/[\w-]+')
        self.github_pattern = re.compile(r'github\.com/[\w-]+')
        
        # Skills database (expandable)
        self.skills_database = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'go', 'rust', 'kotlin'],
            'web': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask'],
            'data': ['sql', 'mongodb', 'postgresql', 'mysql', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform'],
            'tools': ['git', 'jira', 'confluence', 'slack', 'figma', 'photoshop', 'excel', 'powerbi', 'tableau']
        }
    
    def extract_personal_info(self, text: str) -> PersonalInfo:
        """Extract personal information using regex and NLP"""
        personal_info = PersonalInfo()
        
        # Extract email
        email_match = self.email_pattern.search(text)
        if email_match:
            personal_info.email = email_match.group()
        
        # Extract phone
        phone_match = self.phone_pattern.search(text)
        if phone_match:
            personal_info.phone = phone_match.group()
        
        # Extract LinkedIn
        linkedin_match = self.linkedin_pattern.search(text)
        if linkedin_match:
            personal_info.linkedin = linkedin_match.group()
        
        # Extract GitHub
        github_match = self.github_pattern.search(text)
        if github_match:
            personal_info.github = github_match.group()
        
        # Extract name using NER if spaCy is available
        if self.nlp:
            doc = self.nlp(text[:500])  # Process first 500 chars for efficiency
            for ent in doc.ents:
                if ent.label_ == "PERSON" and not personal_info.name:
                    personal_info.name = ent.text
                    break
        
        return personal_info
    
    def extract_experience(self, text: str) -> List[Experience]:
        """Extract work experience using pattern matching and NLP"""
        experiences = []
        
        # Define experience section patterns
        exp_patterns = [
            r'(?i)(work\s+experience|experience|employment\s+history|professional\s+experience)',
            r'(?i)(career\s+summary|work\s+history)'
        ]
        
        # Find experience section
        exp_section = ""
        for pattern in exp_patterns:
            match = re.search(pattern, text)
            if match:
                start_idx = match.end()
                # Look for next major section
                next_section = re.search(r'(?i)(education|skills|projects|certifications)', text[start_idx:])
                end_idx = next_section.start() + start_idx if next_section else len(text)
                exp_section = text[start_idx:end_idx]
                break
        
        if exp_section:
            # Split by job entries (look for date patterns)
            date_pattern = r'(\d{4}[-\s]*\d{0,4}|\w+\s+\d{4})'
            job_entries = re.split(date_pattern, exp_section)
            
            for i in range(1, len(job_entries), 2):  # Skip first empty split
                if i + 1 < len(job_entries):
                    duration = job_entries[i]
                    description = job_entries[i + 1].strip()
                    
                    if len(description) > 20:  # Filter out noise
                        exp = Experience()
                        exp.duration = duration
                        exp.description = description[:200]  # Limit description length
                        
                        # Extract job title and company (simple heuristic)
                        lines = description.split('\n')[:3]
                        for line in lines:
                            line = line.strip()
                            if line and not exp.title:
                                exp.title = line
                            elif line and not exp.company and line != exp.title:
                                exp.company = line
                                break
                        
                        experiences.append(exp)
        
        return experiences[:5]  # Limit to 5 most recent experiences
    
    def extract_education(self, text: str) -> List[Education]:
        """Extract education information"""
        educations = []
        
        # Find education section
        edu_pattern = r'(?i)(education|academic\s+background|qualifications)'
        match = re.search(edu_pattern, text)
        
        if match:
            start_idx = match.end()
            next_section = re.search(r'(?i)(experience|skills|projects|certifications)', text[start_idx:])
            end_idx = next_section.start() + start_idx if next_section else len(text)
            edu_section = text[start_idx:end_idx]
            
            # Look for degree patterns
            degree_patterns = [
                r'(Bachelor|Master|PhD|B\.S\.|M\.S\.|B\.A\.|M\.A\.|MBA)',
                r'(Associates|Diploma|Certificate)'
            ]
            
            for pattern in degree_patterns:
                matches = re.finditer(pattern, edu_section, re.IGNORECASE)
                for match in matches:
                    edu = Education()
                    context = edu_section[max(0, match.start()-50):match.end()+100]
                    
                    # Extract degree
                    edu.degree = match.group()
                    
                    # Extract year
                    year_match = re.search(r'(19|20)\d{2}', context)
                    if year_match:
                        edu.year = year_match.group()
                    
                    # Extract institution (heuristic: capitalize words near degree)
                    words = context.split()
                    for i, word in enumerate(words):
                        if word.lower() in edu.degree.lower():
                            # Look for capitalized words after degree
                            for j in range(i+1, min(i+6, len(words))):
                                if words[j][0].isupper() and len(words[j]) > 2:
                                    if not edu.institution:
                                        edu.institution = words[j]
                                    elif len(edu.institution) < 50:
                                        edu.institution += " " + words[j]
                            break
                    
                    if edu.degree:
                        educations.append(edu)
        
        return educations[:3]  # Limit to 3 education entries
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills using pattern matching and skill database"""
        skills = set()
        text_lower = text.lower()
        
        # Extract from skills database
        for category, skill_list in self.skills_database.items():
            for skill in skill_list:
                if skill in text_lower:
                    skills.add(skill.title())
        
        # Extract from skills section if exists
        skills_pattern = r'(?i)(skills|technical\s+skills|competencies)'
        match = re.search(skills_pattern, text)
        
        if match:
            start_idx = match.end()
            next_section = re.search(r'(?i)(experience|education|projects|certifications)', text[start_idx:])
            end_idx = next_section.start() + start_idx if next_section else min(start_idx + 300, len(text))
            skills_section = text[start_idx:end_idx]
            
            # Extract comma-separated skills
            potential_skills = re.split(r'[,\n•\-\|]', skills_section)
            for skill in potential_skills:
                skill = skill.strip()
                if 2 < len(skill) < 30 and not any(char.isdigit() for char in skill):
                    skills.add(skill.title())
        
        return list(skills)[:20]  # Limit to 20 skills
    
    def generate_summary(self, text: str) -> str:
        """Generate a summary of the resume"""
        if self.nlp:
            doc = self.nlp(text[:1000])  # Process first 1000 chars
            sentences = [sent.text for sent in doc.sents]
            
            # Score sentences based on position and content
            scored_sentences = []
            for i, sentence in enumerate(sentences[:10]):
                score = 1.0 / (i + 1)  # Position score
                if any(word in sentence.lower() for word in ['experience', 'skilled', 'professional', 'expertise']):
                    score += 0.5
                scored_sentences.append((sentence, score))
            
            # Select top sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            summary = ' '.join([sent[0] for sent in scored_sentences[:2]])
            return summary[:300]  # Limit summary length
        
        # Fallback: take first few sentences
        sentences = sent_tokenize(text)
        return ' '.join(sentences[:2])[:300]

class BiasDetector:
    """Detect potential bias in resumes and job matching"""
    
    def __init__(self):
        self.bias_indicators = {
            'gender': ['he/him', 'she/her', 'male', 'female', 'woman', 'man', 'wife', 'husband'],
            'age': ['years old', 'born in', 'age', 'retired', 'senior', 'young', 'fresh graduate'],
            'ethnicity': ['african', 'asian', 'hispanic', 'latino', 'caucasian', 'white', 'black'],
            'religion': ['christian', 'muslim', 'jewish', 'hindu', 'buddhist', 'church', 'mosque', 'temple'],
            'location': ['from', 'native', 'immigrant', 'foreign', 'international student']
        }
    
    def detect_bias(self, text: str) -> Dict[str, float]:
        """Detect bias indicators in resume text"""
        bias_scores = {}
        text_lower = text.lower()
        
        for category, indicators in self.bias_indicators.items():
            score = 0
            for indicator in indicators:
                if indicator in text_lower:
                    score += 1
            
            # Normalize score
            bias_scores[category] = min(score / len(indicators), 1.0)
        
        return bias_scores
    
    def calculate_fairness_score(self, bias_scores: Dict[str, float]) -> float:
        """Calculate overall fairness score (lower is better)"""
        if not bias_scores:
            return 0.0
        
        total_bias = sum(bias_scores.values())
        return min(total_bias / len(bias_scores), 1.0)

class ResumeQualityAnalyzer:
    """Analyze resume quality and provide scoring"""
    
    def __init__(self):
        self.quality_factors = {
            'length': {'min': 200, 'max': 2000, 'weight': 0.15},
            'sections': {'required': ['experience', 'education', 'skills'], 'weight': 0.25},
            'contact_info': {'required': ['email'], 'weight': 0.20},
            'formatting': {'weight': 0.15},
            'keyword_density': {'weight': 0.25}
        }
    
    def analyze_quality(self, resume_data: ResumeData) -> float:
        """Analyze resume quality and return score (0-1)"""
        scores = []
        
        # Length score
        text_length = len(resume_data.raw_text)
        length_score = self._calculate_length_score(text_length)
        scores.append(length_score * self.quality_factors['length']['weight'])
        
        # Sections score
        sections_score = self._calculate_sections_score(resume_data)
        scores.append(sections_score * self.quality_factors['sections']['weight'])
        
        # Contact info score
        contact_score = self._calculate_contact_score(resume_data.personal_info)
        scores.append(contact_score * self.quality_factors['contact_info']['weight'])
        
        # Formatting score (basic heuristic)
        formatting_score = self._calculate_formatting_score(resume_data.raw_text)
        scores.append(formatting_score * self.quality_factors['formatting']['weight'])
        
        # Keyword density score
        keyword_score = self._calculate_keyword_score(resume_data)
        scores.append(keyword_score * self.quality_factors['keyword_density']['weight'])
        
        return sum(scores)
    
    def _calculate_length_score(self, length: int) -> float:
        """Calculate score based on resume length"""
        min_len = self.quality_factors['length']['min']
        max_len = self.quality_factors['length']['max']
        
        if length < min_len:
            return length / min_len
        elif length > max_len:
            return max(0.5, 1.0 - (length - max_len) / max_len)
        else:
            return 1.0
    
    def _calculate_sections_score(self, resume_data: ResumeData) -> float:
        """Calculate score based on required sections"""
        required_sections = self.quality_factors['sections']['required']
        present_sections = 0
        
        if resume_data.experience:
            present_sections += 1
        if resume_data.education:
            present_sections += 1
        if resume_data.skills:
            present_sections += 1
        
        return present_sections / len(required_sections)
    
    def _calculate_contact_score(self, personal_info: PersonalInfo) -> float:
        """Calculate score based on contact information"""
        score = 0
        if personal_info.email:
            score += 0.5
        if personal_info.phone:
            score += 0.3
        if personal_info.linkedin:
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_formatting_score(self, text: str) -> float:
        """Calculate basic formatting score"""
        score = 0
        
        # Check for proper capitalization
        sentences = sent_tokenize(text)
        properly_capitalized = sum(1 for sent in sentences if sent and sent[0].isupper())
        if sentences:
            score += 0.3 * (properly_capitalized / len(sentences))
        
        # Check for bullet points or structure
        if '•' in text or '-' in text or re.search(r'\n\s*\d+\.', text):
            score += 0.4
        
        # Check for section headers
        if re.search(r'(?i)(experience|education|skills)', text):
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_keyword_score(self, resume_data: ResumeData) -> float:
        """Calculate score based on relevant keywords"""
        important_keywords = [
            'experience', 'skills', 'projects', 'achieved', 'managed', 'developed',
            'implemented', 'improved', 'led', 'created', 'designed', 'analyzed'
        ]
        
        text_lower = resume_data.raw_text.lower()
        keyword_count = sum(1 for keyword in important_keywords if keyword in text_lower)
        
        return min(keyword_count / len(important_keywords), 1.0)

class JobMatcher:
    """Match resumes to job descriptions"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def calculate_match_score(self, resume_text: str, job_description: str) -> float:
        """Calculate similarity score between resume and job description"""
        try:
            # Combine texts for vectorization
            texts = [resume_text, job_description]
            
            # Create TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            
            return float(similarity[0][0])
        except Exception as e:
            print(f"Error in match calculation: {e}")
            return 0.0

class ResumeProcessor:
    """Main resume processing class"""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.nlp_processor = AdvancedNLPProcessor()
        self.bias_detector = BiasDetector()
        self.quality_analyzer = ResumeQualityAnalyzer()
        self.job_matcher = JobMatcher()
    
    def process_resume(self, file_content: bytes, filename: str, job_description: str = "") -> ResumeData:
        """Process a resume file and extract structured data"""
        
        # Extract text based on file type
        if filename.lower().endswith('.pdf'):
            raw_text = self.doc_processor.extract_text_from_pdf(file_content)
        elif filename.lower().endswith(('.docx', '.doc')):
            raw_text = self.doc_processor.extract_text_from_docx(file_content)
        else:
            raw_text = file_content.decode('utf-8', errors='ignore')
        
        # Extract structured information
        personal_info = self.nlp_processor.extract_personal_info(raw_text)
        experience = self.nlp_processor.extract_experience(raw_text)
        education = self.nlp_processor.extract_education(raw_text)
        skills = self.nlp_processor.extract_skills(raw_text)
        summary = self.nlp_processor.generate_summary(raw_text)
        
        # Create resume data object
        resume_data = ResumeData(
            personal_info=personal_info,
            experience=experience,
            education=education,
            skills=skills,
            summary=summary,
            raw_text=raw_text
        )
        
        # Analyze bias
        bias_scores = self.bias_detector.detect_bias(raw_text)
        resume_data.bias_score = bias_scores
        
        # Calculate quality score
        quality_score = self.quality_analyzer.analyze_quality(resume_data)
        resume_data.quality_score = quality_score
        
        # Calculate job match score if job description provided
        if job_description:
            match_score = self.job_matcher.calculate_match_score(raw_text, job_description)
            resume_data.match_score = match_score
        
        return resume_data

class ATSIntegration:
    """Integration with Applicant Tracking Systems"""
    
    def __init__(self):
        self.supported_formats = ['json', 'xml', 'csv']
    
    def export_to_ats(self, resume_data: ResumeData, format_type: str = 'json') -> str:
        """Export resume data to ATS-compatible format"""
        if format_type == 'json':
            return self._export_json(resume_data)
        elif format_type == 'xml':
            return self._export_xml(resume_data)
        elif format_type == 'csv':
            return self._export_csv(resume_data)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _export_json(self, resume_data: ResumeData) -> str:
        """Export to JSON format"""
        data = {
            'personal_info': asdict(resume_data.personal_info),
            'experience': [asdict(exp) for exp in resume_data.experience],
            'education': [asdict(edu) for edu in resume_data.education],
            'skills': resume_data.skills,
            'summary': resume_data.summary,
            'quality_score': resume_data.quality_score,
            'match_score': resume_data.match_score,
            'bias_analysis': resume_data.bias_score
        }
        return json.dumps(data, indent=2)
    
    def _export_xml(self, resume_data: ResumeData) -> str:
        """Export to XML format"""
        # Simplified XML export
        xml = "<resume>\n"
        xml += f"  <name>{resume_data.personal_info.name}</name>\n"
        xml += f"  <email>{resume_data.personal_info.email}</email>\n"
        xml += f"  <phone>{resume_data.personal_info.phone}</phone>\n"
        xml += f"  <skills>{', '.join(resume_data.skills)}</skills>\n"
        xml += f"  <quality_score>{resume_data.quality_score}</quality_score>\n"
        xml += "</resume>"
        return xml
    
    def _export_csv(self, resume_data: ResumeData) -> str:
        """Export to CSV format"""
        import io
        output = io.StringIO()
        
        # Write header
        output.write("Name,Email,Phone,Skills,Quality Score,Match Score\n")
        
        # Write data
        skills_str = "; ".join(resume_data.skills)
        output.write(f'"{resume_data.personal_info.name}","{resume_data.personal_info.email}","{resume_data.personal_info.phone}","{skills_str}",{resume_data.quality_score},{resume_data.match_score}\n')
        
        return output.getvalue()

# Flask Web Application
app = Flask(__name__)
CORS(app)

# Global processor instance
processor = ResumeProcessor()
ats_integration = ATSIntegration()

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/process_resume', methods=['POST'])
def process_resume_api():
    """API endpoint to process resume"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        job_description = request.form.get('job_description', '')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Process the resume
        file_content = file.read()
        resume_data = processor.process_resume(file_content, file.filename, job_description)
        
        # Convert to dict for JSON response
        result = {
            'personal_info': asdict(resume_data.personal_info),
            'experience': [asdict(exp) for exp in resume_data.experience],
            'education': [asdict(edu) for edu in resume_data.education],
            'skills': resume_data.skills,
            'summary': resume_data.summary,
            'quality_score': round(resume_data.quality_score, 2),
            'match_score': round(resume_data.match_score, 2),
            'bias_analysis': resume_data.bias_score,
            'fairness_score': round(processor.bias_detector.calculate_fairness_score(resume_data.bias_score), 2)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/<format_type>')
def export_resume(format_type):
    """Export resume data to ATS format"""
    try:
        # This would normally retrieve the processed resume data
        # For demo, we'll return a sample
        sample_data = ResumeData(
            personal_info=PersonalInfo(name="John Doe", email="john@example.com"),
            experience=[],
            education=[],
            skills=["Python", "Machine Learning"],
            summary="Experienced developer"
        )
        
        exported_data = ats_integration.export_to_ats(sample_data, format_type)
        return exported_data, 200, {'Content-Type': 'text/plain'}
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_process', methods=['POST'])
def batch_process():
    """Process multiple resumes"""
    try:
        files = request.files.getlist('files')
        job_description = request.form.get('job_description', '')
        
        results = []
        for file in files:
            if file.filename:
                file_content = file.read()
                resume_data = processor.process_resume(file_content, file.filename, job_description)
                
                result = {
                    'filename': file.filename,
                    'name': resume_data.personal_info.name,
                    'email': resume_data.personal_info.email,
                    'quality_score': round(resume_data.quality_score, 2),
                    'match_score': round(resume_data.match_score, 2),
                    'skills_count': len(resume_data.skills),
                    'bias_analysis': resume_data.bias_score
                }
                results.append(result)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# HTML Templates (inline for demo)
dashboard_html = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Resume Processing System</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .upload-area { border: 2px dashed #007bff; padding: 30px; text-align: center; margin: 20px 0; border-radius: 10px; background: #f8f9ff; }
        .upload-area:hover { background: #e6f2ff; }
        .results { margin-top: 20px; }
        .section { margin: 20px 0; padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; background: #fafafa; }
        .score { font-size: 20px; font-weight: bold; margin: 10px 0; }
        .score.high { color: #28a745; }
        .score.medium { color: #ffc107; }
        .score.low { color: #dc3545; }
        .skills { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }
        .skill-tag { background: #007bff; color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px; }
        .bias-indicator { display: inline-block; padding: 3px 8px; border-radius: 10px; font-size: 11px; margin: 2px; }
        .bias-low { background: #d4edda; color: #155724; }
        .bias-medium { background: #fff3cd; color: #856404; }
        .bias-high { background: #f8d7da; color: #721c24; }
        .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
        .btn:hover { background: #0056b3; }
        .btn-secondary { background: #6c757d; }
        .btn-secondary:hover { background: #545b62; }
        .progress-bar { width: 100%; height: 20px; background: #e0e0e0; border-radius: 10px; overflow: hidden; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #28a745, #ffc107, #dc3545); transition: width 0.3s; }
        .tabs { display: flex; border-bottom: 2px solid #e0e0e0; margin-bottom: 20px; }
        .tab { padding: 10px 20px; cursor: pointer; border-bottom: 2px solid transparent; }
        .tab.active { border-bottom-color: #007bff; background: #f8f9ff; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AI Resume Processing System</h1>
            <p>Advanced NLP-powered resume analysis with bias detection and ATS integration</p>
        </div>
        
        <div class="tabs">
            <div class="tab active" onclick="switchTab('single')">Single Resume</div>
            <div class="tab" onclick="switchTab('batch')">Batch Processing</div>
            <div class="tab" onclick="switchTab('analytics')">Analytics</div>
        </div>
        
        <div id="single" class="tab-content active">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <p>📄 Click to upload resume (PDF, DOCX, TXT)</p>
                <input type="file" id="fileInput" style="display: none;" accept=".pdf,.docx,.doc,.txt" onchange="handleFileUpload(this)">
            </div>
            
            <div class="section">
                <h3>Job Description (Optional)</h3>
                <textarea id="jobDescription" placeholder="Paste job description here for matching analysis..." 
                         style="width: 100%; height: 100px; padding: 10px; border: 1px solid #ddd; border-radius: 5px;"></textarea>
            </div>
            
            <div id="results" class="results" style="display: none;">
                <div class="grid">
                    <div class="section">
                        <h3>📋 Personal Information</h3>
                        <div id="personalInfo"></div>
                    </div>
                    
                    <div class="section">
                        <h3>📊 Quality & Match Scores</h3>
                        <div id="scores"></div>
                    </div>
                </div>
                
                <div class="section">
                    <h3>🎯 Skills</h3>
                    <div id="skills" class="skills"></div>
                </div>
                
                <div class="section">
                    <h3>💼 Experience</h3>
                    <div id="experience"></div>
                </div>
                
                <div class="section">
                    <h3>🎓 Education</h3>
                    <div id="education"></div>
                </div>
                
                <div class="section">
                    <h3>⚖️ Bias Analysis</h3>
                    <div id="biasAnalysis"></div>
                </div>
                
                <div class="section">
                    <h3>📄 Summary</h3>
                    <div id="summary"></div>
                </div>
                
                <div class="section">
                    <h3>🔗 Export to ATS</h3>
                    <button class="btn" onclick="exportData('json')">Export JSON</button>
                    <button class="btn btn-secondary" onclick="exportData('xml')">Export XML</button>
                    <button class="btn btn-secondary" onclick="exportData('csv')">Export CSV</button>
                </div>
            </div>
        </div>
        
        <div id="batch" class="tab-content">
            <div class="upload-area" onclick="document.getElementById('batchFileInput').click()">
                <p>📁 Click to upload multiple resumes</p>
                <input type="file" id="batchFileInput" style="display: none;" multiple accept=".pdf,.docx,.doc,.txt" onchange="handleBatchUpload(this)">
            </div>
            
            <div class="section">
                <h3>Job Description</h3>
                <textarea id="batchJobDescription" placeholder="Job description for batch matching..." 
                         style="width: 100%; height: 80px; padding: 10px; border: 1px solid #ddd; border-radius: 5px;"></textarea>
            </div>
            
            <div id="batchResults" style="display: none;">
                <div class="section">
                    <h3>📊 Batch Processing Results</h3>
                    <div id="batchTable"></div>
                </div>
            </div>
        </div>
        
        <div id="analytics" class="tab-content">
            <div class="section">
                <h3>📈 System Analytics</h3>
                <p>Analytics dashboard would show:</p>
                <ul>
                    <li>Processing statistics</li>
                    <li>Quality score distributions</li>
                    <li>Bias detection trends</li>
                    <li>Skills frequency analysis</li>
                    <li>ATS integration logs</li>
                </ul>
            </div>
        </div>
    </div>

    <script>
        function switchTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
        }
        
        function handleFileUpload(input) {
            if (input.files && input.files[0]) {
                const file = input.files[0];
                const jobDesc = document.getElementById('jobDescription').value;
                
                const formData = new FormData();
                formData.append('file', file);
                formData.append('job_description', jobDesc);
                
                // Show loading
                document.getElementById('results').style.display = 'block';
                document.getElementById('personalInfo').innerHTML = '<p>⏳ Processing resume...</p>';
                
                fetch('/api/process_resume', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        alert('Error: ' + data.error);
                        return;
                    }
                    displayResults(data);
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Error processing resume');
                });
            }
        }
        
        function displayResults(data) {
            // Personal Info
            const personalInfo = data.personal_info;
            document.getElementById('personalInfo').innerHTML = `
                <p><strong>Name:</strong> ${personalInfo.name || 'Not found'}</p>
                <p><strong>Email:</strong> ${personalInfo.email || 'Not found'}</p>
                <p><strong>Phone:</strong> ${personalInfo.phone || 'Not found'}</p>
                <p><strong>LinkedIn:</strong> ${personalInfo.linkedin || 'Not found'}</p>
                <p><strong>GitHub:</strong> ${personalInfo.github || 'Not found'}</p>
            `;
            
            // Scores
            const qualityClass = data.quality_score > 0.7 ? 'high' : data.quality_score > 0.4 ? 'medium' : 'low';
            const matchClass = data.match_score > 0.7 ? 'high' : data.match_score > 0.4 ? 'medium' : 'low';
            
            document.getElementById('scores').innerHTML = `
                <div class="score ${qualityClass}">Quality Score: ${(data.quality_score * 100).toFixed(1)}%</div>
                <div class="progress-bar"><div class="progress-fill" style="width: ${data.quality_score * 100}%"></div></div>
                <div class="score ${matchClass}">Match Score: ${(data.match_score * 100).toFixed(1)}%</div>
                <div class="progress-bar"><div class="progress-fill" style="width: ${data.match_score * 100}%"></div></div>
                <div class="score">Fairness Score: ${(data.fairness_score * 100).toFixed(1)}%</div>
            `;
            
            // Skills
            const skillsHtml = data.skills.map(skill => 
                `<span class="skill-tag">${skill}</span>`
            ).join('');
            document.getElementById('skills').innerHTML = skillsHtml || '<p>No skills detected</p>';
            
            // Experience
            const experienceHtml = data.experience.map(exp => `
                <div style="margin: 10px 0; padding: 10px; border-left: 3px solid #007bff;">
                    <h4>${exp.title || 'Position'} at ${exp.company || 'Company'}</h4>
                    <p><strong>Duration:</strong> ${exp.duration}</p>
                    <p>${exp.description}</p>
                </div>
            `).join('');
            document.getElementById('experience').innerHTML = experienceHtml || '<p>No experience found</p>';
            
            // Education
            const educationHtml = data.education.map(edu => `
                <div style="margin: 10px 0; padding: 10px; border-left: 3px solid #28a745;">
                    <h4>${edu.degree}</h4>
                    <p><strong>Institution:</strong> ${edu.institution}</p>
                    <p><strong>Year:</strong> ${edu.year}</p>
                </div>
            `).join('');
            document.getElementById('education').innerHTML = educationHtml || '<p>No education found</p>';
            
            // Bias Analysis
            const biasHtml = Object.entries(data.bias_analysis).map(([category, score]) => {
                const level = score > 0.6 ? 'high' : score > 0.3 ? 'medium' : 'low';
                return `<span class="bias-indicator bias-${level}">${category}: ${(score * 100).toFixed(1)}%</span>`;
            }).join('');
            document.getElementById('biasAnalysis').innerHTML = biasHtml;
            
            // Summary
            document.getElementById('summary').innerHTML = `<p>${data.summary}</p>`;
        }
        
        function handleBatchUpload(input) {
            if (input.files && input.files.length > 0) {
                const files = Array.from(input.files);
                const jobDesc = document.getElementById('batchJobDescription').value;
                
                const formData = new FormData();
                files.forEach(file => formData.append('files', file));
                formData.append('job_description', jobDesc);
                
                document.getElementById('batchResults').style.display = 'block';
                document.getElementById('batchTable').innerHTML = '<p>⏳ Processing resumes...</p>';
                
                fetch('/api/batch_process', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        alert('Error: ' + data.error);
                        return;
                    }
                    displayBatchResults(data);
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Error processing resumes');
                });
            }
        }
        
        function displayBatchResults(results) {
            const tableHtml = `
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background: #f8f9fa;">
                            <th style="padding: 10px; border: 1px solid #ddd;">Filename</th>
                            <th style="padding: 10px; border: 1px solid #ddd;">Name</th>
                            <th style="padding: 10px; border: 1px solid #ddd;">Email</th>
                            <th style="padding: 10px; border: 1px solid #ddd;">Quality</th>
                            <th style="padding: 10px; border: 1px solid #ddd;">Match</th>
                            <th style="padding: 10px; border: 1px solid #ddd;">Skills</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${results.map(result => `
                            <tr>
                                <td style="padding: 10px; border: 1px solid #ddd;">${result.filename}</td>
                                <td style="padding: 10px; border: 1px solid #ddd;">${result.name || 'N/A'}</td>
                                <td style="padding: 10px; border: 1px solid #ddd;">${result.email || 'N/A'}</td>
                                <td style="padding: 10px; border: 1px solid #ddd;">${(result.quality_score * 100).toFixed(1)}%</td>
                                <td style="padding: 10px; border: 1px solid #ddd;">${(result.match_score * 100).toFixed(1)}%</td>
                                <td style="padding: 10px; border: 1px solid #ddd;">${result.skills_count}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
            document.getElementById('batchTable').innerHTML = tableHtml;
        }
        
        function exportData(format) {
            window.open(`/api/export/${format}`, '_blank');
        }
    </script>
</body>
</html>
"""

# Template registration
@app.route('/dashboard.html')
def dashboard():
    return dashboard_html

# Deep Learning Enhancement Module
class DeepLearningProcessor:
    """Enhanced processing using deep learning models"""
    
    def __init__(self):
        # In a real implementation, you would load pre-trained models
        # For demo, we'll simulate advanced processing
        self.embedding_model = None  # Would load sentence transformers
        self.classification_model = None  # Would load custom trained model
        
    def extract_advanced_features(self, text: str) -> Dict[str, Any]:
        """Extract advanced features using deep learning"""
        features = {}
        
        # Simulate semantic embeddings
        features['semantic_score'] = np.random.random()
        
        # Simulate skill level prediction
        features['skill_levels'] = {
            'technical': np.random.random(),
            'communication': np.random.random(),
            'leadership': np.random.random()
        }
        
        # Simulate career progression analysis
        features['career_progression'] = np.random.random()
        
        return features
    
    def predict_job_success(self, resume_data: ResumeData, job_description: str) -> float:
        """Predict job success probability using ML model"""
        # In real implementation, this would use trained models
        # considering factors like experience match, skill relevance, etc.
        
        base_score = resume_data.match_score
        quality_bonus = resume_data.quality_score * 0.2
        experience_bonus = len(resume_data.experience) * 0.05
        
        success_probability = min(base_score + quality_bonus + experience_bonus, 1.0)
        return success_probability

# Enhanced Bias Detection with Fairness Metrics
class AdvancedBiasDetector(BiasDetector):
    """Advanced bias detection with fairness metrics"""
    
    def __init__(self):
        super().__init__()
        self.fairness_metrics = [
            'demographic_parity',
            'equalized_odds',
            'calibration'
        ]
    
    def calculate_fairness_metrics(self, resume_scores: List[Dict]) -> Dict[str, float]:
        """Calculate advanced fairness metrics"""
        metrics = {}
        
        # Simulate fairness calculations
        # In real implementation, these would be calculated based on
        # actual demographic data and scoring distributions
        
        metrics['demographic_parity'] = np.random.random()
        metrics['equalized_odds'] = np.random.random()
        metrics['calibration'] = np.random.random()
        
        return metrics
    
    def generate_bias_report(self, resume_data_list: List[ResumeData]) -> Dict[str, Any]:
        """Generate comprehensive bias analysis report"""
        report = {
            'total_resumes': len(resume_data_list),
            'bias_categories': {},
            'recommendations': []
        }
        
        # Analyze bias across all resumes
        for category in self.bias_indicators.keys():
            scores = [data.bias_score.get(category, 0) for data in resume_data_list if data.bias_score]
            if scores:
                report['bias_categories'][category] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'max': np.max(scores)
                }
        
        # Generate recommendations
        high_bias_categories = [cat for cat, metrics in report['bias_categories'].items() 
                              if metrics['mean'] > 0.5]
        
        if high_bias_categories:
            report['recommendations'].append(
                f"High bias detected in: {', '.join(high_bias_categories)}. "
                "Consider reviewing screening criteria."
            )
        else:
            report['recommendations'].append("Bias levels appear acceptable across categories.")
        
        return report

# Production Deployment Utilities
class ProductionConfig:
    """Configuration for production deployment"""
    
    def __init__(self):
        self.config = {
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'allowed_extensions': {'.pdf', '.docx', '.doc', '.txt'},
            'rate_limit': 100,  # requests per hour
            'cache_timeout': 3600,  # 1 hour
            'batch_size_limit': 50,
            'database_url': os.getenv('DATABASE_URL', 'sqlite:///resumes.db'),
            'redis_url': os.getenv('REDIS_URL', 'redis://localhost:6379'),
            'secret_key': os.getenv('SECRET_KEY', 'dev-key-change-in-production')
        }
    
    def setup_logging(self):
        """Setup production logging"""
        import logging
        from logging.handlers import RotatingFileHandler
        
        if not app.debug:
            file_handler = RotatingFileHandler('logs/resume_processor.log', maxBytes=10240, backupCount=10)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            ))
            file_handler.setLevel(logging.INFO)
            app.logger.addHandler(file_handler)
            app.logger.setLevel(logging.INFO)
            app.logger.info('Resume Processor startup')

# Database Models (using SQLAlchemy for production)
class DatabaseManager:
    """Database management for storing processed resumes"""
    
    def __init__(self):
        # In production, use proper database
        self.resumes_db = []  # Simple in-memory storage for demo
    
    def save_resume(self, resume_data: ResumeData, user_id: str = None) -> str:
        """Save processed resume to database"""
        resume_id = f"resume_{len(self.resumes_db) + 1}"
        
        resume_record = {
            'id': resume_id,
            'user_id': user_id,
            'data': resume_data,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
        
        self.resumes_db.append(resume_record)
        return resume_id
    
    def get_resume(self, resume_id: str) -> Optional[ResumeData]:
        """Retrieve resume by ID"""
        for record in self.resumes_db:
            if record['id'] == resume_id:
                return record['data']
        return None
    
    def search_resumes(self, query: str, filters: Dict = None) -> List[Dict]:
        """Search resumes with filters"""
        results = []
        
        for record in self.resumes_db:
            resume_data = record['data']
            
            # Simple text search
            if query.lower() in resume_data.raw_text.lower():
                result = {
                    'id': record['id'],
                    'name': resume_data.personal_info.name,
                    'email': resume_data.personal_info.email,
                    'quality_score': resume_data.quality_score,
                    'match_score': resume_data.match_score,
                    'skills': resume_data.skills[:5]  # Top 5 skills
                }
                results.append(result)
        
        return results

# Initialize components
db_manager = DatabaseManager()
deep_learning_processor = DeepLearningProcessor()
advanced_bias_detector = AdvancedBiasDetector()
production_config = ProductionConfig()

# Additional API endpoints
@app.route('/api/search_resumes', methods=['POST'])
def search_resumes():
    """Search processed resumes"""
    try:
        query = request.json.get('query', '')
        filters = request.json.get('filters', {})
        
        results = db_manager.search_resumes(query, filters)
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bias_report', methods=['POST'])
def generate_bias_report():
    """Generate bias analysis report"""
    try:
        # Get all processed resumes
        resume_data_list = [record['data'] for record in db_manager.resumes_db]
        
        if not resume_data_list:
            return jsonify({'error': 'No resumes found for analysis'}), 400
        
        report = advanced_bias_detector.generate_bias_report(resume_data_list)
        return jsonify(report)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/advanced_analysis', methods=['POST'])
def advanced_analysis():
    """Perform advanced deep learning analysis"""
    try:
        resume_id = request.json.get('resume_id')
        job_description = request.json.get('job_description', '')
        
        resume_data = db_manager.get_resume(resume_id)
        if not resume_data:
            return jsonify({'error': 'Resume not found'}), 404
        
        # Perform advanced analysis
        advanced_features = deep_learning_processor.extract_advanced_features(resume_data.raw_text)
        success_probability = deep_learning_processor.predict_job_success(resume_data, job_description)
        
        result = {
            'advanced_features': advanced_features,
            'success_probability': success_probability,
            'recommendations': [
                "Consider highlighting technical skills more prominently",
                "Add quantifiable achievements to experience descriptions",
                "Include relevant certifications for the target role"
            ]
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    
    # Setup production configuration
    production_config.setup_logging()
    
    print("🚀 AI Resume Processing System Starting...")
    print("📊 Features enabled:")
    print("   • Advanced NLP processing")
    print("   • Bias detection and fairness metrics")
    print("   • Deep learning enhancements")
    print("   • ATS integration")
    print("   • Batch processing")
    print("   • Web interface")
    print("\n🌐 Access the system at: http://localhost:8000")
    print("📖 API Documentation:")
    print("   • POST /api/process_resume - Process single resume")
    print("   • POST /api/batch_process - Process multiple resumes")
    print("   • POST /api/search_resumes - Search processed resumes")
    print("   • POST /api/bias_report - Generate bias analysis report")
    print("   • GET /api/export/<format> - Export to ATS format")
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=8000)
