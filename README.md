# AI Interview Prep Platform

An intelligent AI-powered interview preparation system that generates customized interview questions based on job descriptions using Google's Gemini AI and LangChain.

## Overview

This platform uses an **End-to-End Retrieval-Augmented Generation (RAG)** pipeline—including document loading, chunking, embedding, and generation—to create personalized interview questions tailored to specific job requirements. It uses an in-memory Chroma vector database to combine pre-existing questions from a crawled repository with dynamically generated AI context.

## Features

- **End-to-End RAG Pipeline**: Implements the full RAG lifecycle (Loading, Chunking, Embedding, Retrieving, Generating).
- **AI-Powered Job Description Parser**: Extracts key information (skills, experience, responsibilities) from unstructured job descriptions.
- **Local Vector Database**: Uses **ChromaDB** for fast, local semantic search without requiring additional API limits.
- **Dynamic Question Generation**: Creates custom interview questions tailored to specific roles and experience levels based on the retrieved context.
- **Interactive Interview Mode**: Step-by-step question presentation with hidden evaluation rubrics.
- **Web Scraping**: Automated question bank building from structured HTML sources.

## Architecture

### Core Components

1. **`crawler.py`** - Question Bank Builder
   - Scrapes interview questions from web sources
   - Stores questions with metadata (type, difficulty, tags)
   - Outputs to `questions_crawled.json`

2. **`parser.py`** - Job Description Analyzer Demo
   - Parses unstructured job descriptions
   - Extracts structured data using Gemini AI via Pydantic schemas

3. **`main.py`** - Main Orchestrator (RAG Pipeline)
   - Executes the end-to-end RAG pipeline
   - Chunks question banks using `RecursiveCharacterTextSplitter`
   - Embeds and stores context in ChromaDB
   - Generates questions by augmenting prompts with retrieved context
   - Provides an interactive interview experience

### Technology Stack

- **LangChain**: AI chain orchestration and RAG pipeline management
- **Google Gemini AI**: Large language models (`gemini-2.5-flash` / `gemini-2.5-pro`) for parsing, generation, and embeddings
- **ChromaDB**: Local vector store for document embeddings and semantic search
- **Pydantic**: Data validation and strict schema enforcement
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
   pip install langchain langchain-google-genai pydantic python-dotenv beautifulsoup4 requests chromadb langchain-text-splitters langchain-community
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

2. **Run the RAG interview generator**
   ```bash
   python main.py
   ```

3. **Test your API key** (optional)
   ```bash
   python test_api.py
   ```

## How It Works

### 4-Step RAG Pipeline

```
[1/4] Parse Job Description
   ↓ Extract structured data (skills, experience, etc.) using Gemini LLM.
   
[2/4] Prepare Vector DB (Load, Chunk, Embed)
   ↓ Parse `questions_crawled.json` into LangChain Documents.
   ↓ Split documents using RecursiveCharacterTextSplitter.
   ↓ Embed chunks and store them in an in-memory ChromaDB.
   
[3/4] Execute RAG (Retrieve + Generate)
   ↓ Retrieve the most relevant chunked context for the highly ranked required skills.
   ↓ Pass enriched context to the Generator Chain to construct highly accurate, tailored questions.
   
[4/4] Interactive Demo
   ↓ Present the AI-generated interview package to the user sequentially, revealing rubrics on demand.
```

## Data Models

### Job Description Schema
```python
- job_title: str
- required_skills: List[str]
- preferred_skills: List[str]
- years_of_experience: int
```

### Generated Question Schema
```python
- type: str  # Behavioral, Technical, System Design
- difficulty: str  # Easy, Medium, Hard
- question: str
- rubric: str  # Evaluation criteria
```

## 🔧 Configuration

### Customizing the Job Description

Edit the `jd_file_path` variable in `main.py` to point to a different text file (e.g., `jd_mock_2.txt`):

```python
jd_file_path = "jd_mock_2.txt"
```

### Adjusting Model Parameters

You can modify the AI models and parameters in the chain functions:

```python
# In main.py
def get_parser_chain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # Change model here
        temperature=0,             # Adjust creativity (0-1)
    )
```

### Available Models
- `gemini-2.5-pro` - Best for complex reasoning
- `gemini-2.5-flash` - Faster, lighter version
- `gemini-1.5-flash` - Older legacy lightweight model

## 📁 Project Structure

```
AI-Interview-Prep-Platform/
│
├── main.py                   # Main RAG orchestrator
├── crawler.py                # Web scraper for question bank baseline
├── parser.py                 # Job description parser implementation
├── questions_crawled.json    # Scraped question database
├── jd_mock_1.txt             # Sample Job Description 1
├── jd_mock_2.txt             # Sample Job Description 2
├── .env                      # Environment variables
├── README.md                 # This file
│
└── Test/                     # Development/testing files
    ├── demo.py               # Earlier matching build
    ├── demo3.py              # Intermediate development iteration
    └── questions.json        # Test question bank
```

## Troubleshooting

1. **Model Not Found Error**
   - Ensure you are using `gemini-2.5-flash` or newer. Some keys do not have access to older 1.5 versions.
   
2. **Pydantic Validation Errors**
   - Ensure you have the `pydantic` package installed and are not using the deprecated Langchain wrappers. Use `.model_dump()` instead of `.dict()`.

3. **ChromaDB Issues**
   - Make sure `langchain-community` and `chromadb` are installed. The DB runs in-memory and regenerates on each execution.
