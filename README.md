# AI Interview Prep Platform

An intelligent AI-powered interview preparation system that generates customized interview questions based on job descriptions using Google's Gemini AI and LangChain.

## Overview

This platform uses advanced AI techniques including **Retrieval-Augmented Generation (RAG)**, **semantic search**, and **structured output generation** to create personalized interview questions tailored to specific job requirements. It combines pre-existing questions from a crawled database with dynamically generated questions to provide comprehensive interview preparation.

## Features

- **AI-Powered Job Description Parser**: Extracts key information (skills, experience, responsibilities) from unstructured job descriptions
- **Intelligent Skill Categorization**: Automatically groups skills into logical categories (Programming Languages, Frameworks, Cloud Platforms, etc.)
- **Semantic Question Matching**: Uses vector embeddings and FAISS to find relevant questions based on meaning, not just keywords
- **Dynamic Question Generation**: Creates custom interview questions tailored to specific roles and experience levels
- **Hybrid Approach**: Combines pre-existing questions with AI-generated content for comprehensive coverage
- **Interactive Interview Mode**: Step-by-step question presentation with evaluation rubrics
- **Web Scraping**: Automated question bank building from structured HTML sources

## Architecture

### Core Components

1. **`crawler.py`** - Question Bank Builder
   - Scrapes interview questions from web sources
   - Stores questions with metadata (type, difficulty, tags)
   - Outputs to `questions_crawled.json`

2. **`parser.py`** - Job Description Analyzer
   - Parses unstructured job descriptions
   - Extracts structured data using Gemini AI
   - Returns: job title, required/preferred skills, experience level, responsibilities

3. **`main.py`** - Main Orchestrator
   - 7-step interview generation pipeline
   - Integrates parsing, matching, and generation
   - Provides interactive interview experience

### Technology Stack

- **LangChain**: AI chain orchestration and prompt management
- **Google Gemini AI**: Large language model for parsing and generation
- **FAISS**: Facebook's vector similarity search library
- **Pydantic**: Data validation and schema enforcement
- **BeautifulSoup**: HTML parsing for web scraping
- **Python-dotenv**: Environment variable management

## Getting Started

### Prerequisites

- Python 3.11+
- Google AI Studio API Key ([Get one here](https://aistudio.google.com/app/apikey))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/surajbijjala1/AI-Interview-Prep-Platform.git
   cd AI-Interview-Prep-Platform
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY='your-google-ai-studio-api-key-here'
   ```

   **Important**: Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### Usage

1. **Build the question bank** (first time only)
   ```bash
   python crawler.py
   ```

2. **Run the interview generator**
   ```bash
   python main.py
   ```

3. **Test your API key** (optional)
   ```bash
   python test_api.py
   ```

## How It Works

### 7-Step Pipeline

```
Step 1: Parse Job Description
   ↓ Extract structured data (skills, experience, etc.)
   
Step 2: Build Skill Graph
   ↓ Categorize skills into logical groups
   
Step 3: Create Vector Store
   ↓ Generate embeddings for semantic search
   
Step 4: Match Existing Questions
   ↓ Find relevant questions from database
   
Step 5: Generate New Questions
   ↓ Create custom questions for specific skills
   
Step 6: Finalize Package
   ↓ Combine matched + generated questions
   
Step 7: Interactive Demo
   ↓ Present questions with evaluation rubrics
```

## Data Models

### Job Description Schema
```python
- job_title: str
- required_skills: List[str]
- preferred_skills: List[str]
- years_of_experience: int
```

### Question Schema
```python
- id: str
- type: str  # Behavioral, Technical, System Design
- difficulty: str  # Easy, Medium, Hard
- question: str
- tags: List[str]
- rubric: str  # Evaluation criteria
```

## 🔧 Configuration

### Customizing the Job Description

Edit the `jd_text` variable in `main.py` (around line 64) to test with different job postings:

```python
jd_text = """
Your job description here...
"""
```

### Adjusting Model Parameters

You can modify the AI models and parameters in the chain functions:

```python
# In main.py
def get_parser_chain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",  # Change model here
        temperature=0,            # Adjust creativity (0-1)
    )
```

### Available Models
- `gemini-1.5-pro` - Best for complex reasoning
- `gemini-1.5-flash` - Faster, lighter version
- `gemini-pro` - Gemini 1.0 (stable)

## 📁 Project Structure

```
AI-Interview-Prep-Platform/
│
├── main.py                    # Main orchestrator (7-step pipeline)
├── crawler.py                 # Web scraper for question bank
├── parser.py                  # Job description parser demo
├── questions_crawled.json    # Scraped question database
├── .env                      # Environment variables (API keys)
├── README.md                 # This file
│
└── Test/                     # Development/testing files
    ├── demo.py               # Earlier version of main
    ├── demo3.py              # Another iteration
    └── questions.json        # Test question bank
```

## Use Cases

- **Job Seekers**: Practice for interviews with role-specific questions
- **Recruiters**: Generate interview questions based on job descriptions
- **Career Coaches**: Create customized interview prep materials
- **Hiring Managers**: Standardize interview processes with AI-generated questions

## Troubleshooting

### Common Issues

1. **API Key Expired Error**
   ```
   Error: 400 API key expired. Please renew the API key.
   ```
   **Solution**: Generate a new API key at [Google AI Studio](https://aistudio.google.com/app/apikey)

2. **Model Not Found Error**
   ```
   Error: 404 models/gemini-pro is not found
   ```
   **Solution**: Run `python test_api.py` to find available models for your API key

3. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'langchain'
   ```
   **Solution**: Make sure you've activated the virtual environment and installed dependencies

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Google Gemini AI for powerful language model capabilities
- LangChain for excellent AI orchestration framework
- FAISS for efficient vector similarity search

## Contact

**Suraj Bijjala**
- GitHub: [@surajbijjala1](https://github.com/surajbijjala1)

---
If you found this project helpful, please consider giving it a star!

**Built with ❤️ using AI and Python**