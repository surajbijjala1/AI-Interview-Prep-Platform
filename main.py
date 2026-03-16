import os
import json
import time
from dotenv import load_dotenv
from typing import List, Dict


# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# --- Load API Key ---
load_dotenv()

# --- 1. DATA MODELS (SCHEMAS) ---
class ParsedJD(BaseModel):
    job_title: str = Field(description="The official title of the job role.")
    required_skills: List[str] = Field(description="A list of essential skills, tools, or technologies required.")
    preferred_skills: List[str] = Field(description="A list of preferred but not mandatory skills.")
    years_of_experience: int = Field(description="The minimum number of years of professional experience required.")

class GeneratedQuestion(BaseModel):
    type: str = Field(description="The type of question (e.g., Behavioral, Technical).")
    difficulty: str = Field(description="The estimated difficulty (Easy, Medium, Hard).")
    question: str = Field(description="The interview question text.")
    rubric: str = Field(description="A brief evaluation rubric for a good answer.")

# --- CORRECTED SKILL GRAPH MODELS ---
# This new structure is more robust and less prone to validation errors.
class SkillCategory(BaseModel):
    """Represents a single category of skills."""
    category_name: str = Field(description="The name of the skill category (e.g., 'Web Framework', 'Database').")
    skills: List[str] = Field(description="The list of skills from the job description that fall under this category.")

class SkillGraph(BaseModel):
    """A model representing a list of categorized skills."""
    graph: List[SkillCategory] = Field(description="A list of skill categories, where each category contains a list of relevant skills.")

# --- 2. AI CHAIN DEFINITIONS ---
def get_parser_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, convert_system_message_to_human=True)
    prompt = ChatPromptTemplate.from_template("From the job description, extract information into the provided schema.\n\nJob Description:\n{job_description}")
    return prompt | llm.with_structured_output(ParsedJD)

def get_generator_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5, convert_system_message_to_human=True)
    prompt_template = """
    You are an expert interviewer for a tech company.
    Based on the following context, generate one high-quality interview question.
    
    Context: {context}
    Job Title: {job_title}
    Required Skill: {skill}
    Years of Experience: {experience}
    
    Ensure the question is relevant for the candidate's experience level.
    Return the question with all required metadata.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    return prompt | llm.with_structured_output(GeneratedQuestion)

# --- 3. RAG PIPELINE FUNCTIONS ---
def prepare_vector_db(file_path: str):
    print(f"\n[RAG - Loading] Loading context from {file_path}...")
    try:
        with open(file_path, "r") as f:
            if file_path.endswith('.json'):
                data = json.load(f)
                documents = [
                    Document(
                        page_content=f"Question: {q['question']}\nTags: {', '.join(q['tags'])}\nType: {q.get('type', '')}\nDifficulty: {q.get('difficulty', '')}",
                        metadata={"id": q.get("id", ""), "source": file_path}
                    ) for q in data
                ]
            else:
                text = f.read()
                documents = [Document(page_content=text, metadata={"source": file_path})]
    except FileNotFoundError:
        print(f"❌ Error: `{file_path}` not found.")
        return None

    print("[RAG - Chunking] Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)

    print("[RAG - Embedding & Vector DB] Creating Chroma vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Using Chroma as the vector store in-memory
    vector_store = Chroma.from_documents(documents=splits, embedding=embeddings, collection_name="interview_context")
    return vector_store

# --- 4. MAIN ORCHESTRATION FUNCTION ---
def main():
    print("🚀 Starting End-to-End RAG Interview Generator...")

    jd_file_path = "jd_mock_1.txt"
    print(f"\n[0/4] Loading Job Description from {jd_file_path}...")
    try:
        with open(jd_file_path, "r") as f:
            jd_text = f.read()
    except FileNotFoundError:
        print(f"❌ Error: `{jd_file_path}` not found.")
        return
    
    print("\n[1/4] Parsing Job Description...")
    parsed_jd = get_parser_chain().invoke({"job_description": jd_text})
    print(f"✅ Done. Role: {parsed_jd.job_title}, Experience: {parsed_jd.years_of_experience} years.")

    # Execute RAG Pipeline components (Load, Chunk, Embed)
    vector_store = prepare_vector_db("questions_crawled.json")
    if not vector_store:
        return
    
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    interview_package = []
    # Take up to 3 required skills to generate questions for
    skills_for_generation = parsed_jd.required_skills[:3]

    print("\n[2/4] Executing RAG (Retrieval + Generation)...")
    generator_chain = get_generator_chain()

    for skill in skills_for_generation:
        print(f"  -> Generating RAG question for '{skill}'...")
        
        # [RAG - Augmenting] Retrieve relevant context from DB
        retrieved_docs = retriever.invoke(skill)
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # [RAG - Generating] Pass enriched context to the LLM
        new_question = generator_chain.invoke({
            "context": context_text,
            "job_title": parsed_jd.job_title,
            "skill": skill,
            "experience": parsed_jd.years_of_experience
        })
        interview_package.append(new_question.model_dump())

    print("✅ Done.")

    print("\n[3/4] Starting Interactive Demo...")
    time.sleep(1)

    print("\n" + "="*50)
    print("🎉 RAG Interview Package Generated")
    print("="*50 + "\n")

    total_questions = len(interview_package)
    for i, question_data in enumerate(interview_package):
        print(f"--- Question {i+1} of {total_questions} ---\n")
        print(f"  Type: {question_data.get('type', 'N/A')}")
        print(f"  Difficulty: {question_data.get('difficulty', 'N/A')}\n")
        print(f"  Question: {question_data['question']}\n")
        
        if 'rubric' in question_data and question_data['rubric']:
            input("  (Press Enter to reveal the evaluation rubric...)")
            print(f"\n  --- Evaluation Rubric ---\n  {question_data['rubric']}\n  -------------------------")
        
        if i < total_questions - 1:
            input("\n(Press Enter for the next question...)")
            print("\n" + "="*50 + "\n")
        else:
            print("\n" + "="*50)
            print("✅ End of Interview. Good luck!")
            print("="*50)

if __name__ == "__main__":
    main()
