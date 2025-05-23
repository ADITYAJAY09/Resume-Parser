# Resume-Parser
AI Resume Processing System

An advanced AI-powered resume processing system built with Flask. This system leverages natural language processing (NLP), machine learning, and deep learning techniques to analyze resumes, detect bias, score resume quality, match resumes to job descriptions, and integrate with Applicant Tracking Systems (ATS). It provides a web dashboard for single and batch resume processing with detailed analytics.

Features

- Extract personal information, experience, education, skills, and summary from resumes (PDF, DOCX, TXT).
- Advanced NLP processing using spaCy, NLTK, and scikit-learn.
- Bias detection across multiple categories (gender, age, ethnicity, religion, location).
- Resume quality scoring based on length, sections, contact info, formatting, and keywords.
- Job description matching with TF-IDF and cosine similarity.
- Batch processing of multiple resumes.
- Export processed data to ATS-compatible formats (JSON, XML, CSV).
- Web dashboard for uploading resumes, viewing analysis, and exporting data.
- Deep learning enhancements for advanced feature extraction and job success prediction.
- Production-ready logging, rate limiting, and configuration.

Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create and activate a Python virtual environment (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

4. Download required NLTK data (the app attempts this automatically, but you can do it manually):

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt_tab')
```

5. (Optional) Download spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

Usage

Run the Flask application:

```bash
python app.py
```

Access the web dashboard at:

```
http://localhost:8000
```

Upload resumes via the dashboard for single or batch processing. Optionally provide a job description for matching analysis.

API Endpoints

- `POST /api/process_resume` - Process a single resume file. Accepts file upload and optional job description.
- `POST /api/batch_process` - Process multiple resumes in batch.
- `GET /api/export/<format>` - Export processed resume data to ATS format (`json`, `xml`, `csv`).
- `POST /api/search_resumes` - Search processed resumes with query and filters.
- `POST /api/bias_report` - Generate bias analysis report across processed resumes.
- `POST /api/advanced_analysis` - Perform advanced deep learning analysis on a resume.

Project Structure

- `app.py` - Main Flask application and core processing logic.
- `templates/` - HTML templates for the web dashboard.
- `logs/` - Log files for application runtime.
- `uploads/` - Directory for uploaded resume files.

Notes

- Supports PDF, DOCX, and plain text resume formats.
- Uses spaCy and NLTK for NLP tasks; ensure models and data are downloaded.
- Designed for extensibility with deep learning and ATS integration modules.
- Logging and production configurations are included for deployment readiness.

License

This project is licensed under the MIT License.

Contact

For questions or contributions, please open an issue or submit a pull request.
