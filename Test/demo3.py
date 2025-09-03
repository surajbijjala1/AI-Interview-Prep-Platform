# main.py

import os
import json
from dotenv import load_dotenv
from typing import List

# --- LangChain Imports ---
# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import FAISS

# --- Load API Key ---
load_dotenv()

# --- 1. PARSER (copied and adapted from Step 1) ---
class ParsedJD(BaseModel):
    job_title: str = Field(description="The official title of the job role.")
    required_skills: List[str] = Field(description="A list of essential skills, tools, or technologies required for the role.")
    preferred_skills: List[str] = Field(description="A list of skills that are preferred but not mandatory.")
    years_of_experience: int = Field(description="The minimum number of years of professional experience required.")

def get_parser_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0, convert_system_message_to_human=True)
    prompt_template = "From the job description, extract information into the provided schema.\n\nJob Description:\n{job_description}"
    prompt = ChatPromptTemplate.from_template(prompt_template)
    return prompt | llm.with_structured_output(ParsedJD)

# --- 2. GENERATOR (for creating new questions) ---
class GeneratedQuestion(BaseModel):
    type: str = Field(description="The type of question (e.g., Behavioral, Technical, System Design).")
    difficulty: str = Field(description="The estimated difficulty (Easy, Medium, Hard).")
    question: str = Field(description="The interview question text.")
    rubric: str = Field(description="A brief evaluation rubric for what to look for in a good answer.")

def get_generator_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0, convert_system_message_to_human=True) # Higher temp for more creative questions
    prompt_template = """
    You are an expert interviewer for a tech company.
    Generate one high-quality interview question based on the following details:
    Job Title: {job_title}
    Required Skill: {skill}
    Years of Experience: {experience}
    
    Ensure the question is relevant for the candidate's experience level.
    Return the question with all required metadata.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    return prompt | llm.with_structured_output(GeneratedQuestion)

# --- 3. MAIN ORCHESTRATION ---
def main():
    print("🚀 Starting Interview Generator...")

    # --- Load Data ---
    with open("questions.json", "r") as f:
        question_bank = json.load(f)

    # For this example, we'll use a sample JD. You can replace this with `input()` or file reading.
    jd_text = """
    Senior Python Developer

    Company: Tech Solutions Inc.
    Location: Remote

    We are seeking a Senior Python Developer with at least 5 years of experience to join our dynamic team. The ideal candidate will be responsible for developing and maintaining our core backend services.

    Required Skills:
    - Python, Django, Flask
    - Experience with RESTful APIs
    - Strong understanding of SQL and database design (PostgreSQL)
    - Git

    Preferred Skills:
    - Docker, Kubernetes
    - Experience with AWS or other cloud platforms
    - Knowledge of microservices architecture
    """
    print("\n[1/4] Parsing Job Description...")
    parser_chain = get_parser_chain()
    parsed_jd = parser_chain.invoke({"job_description": jd_text})
    print(f"✅ Done. Role: {parsed_jd.job_title}, Experience: {parsed_jd.years_of_experience} years.")

    # --- Setup Vector Store for Matching ---
    # 
    # Vector search finds items based on conceptual meaning, not just keywords.
    print("\n[2/4] Creating Vector Store for question matching...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # We create "Documents" where the text includes the question and its tags for better matching.
    documents = [
        Document(page_content=f"Question: {q['question']} Tags: {', '.join(q['tags'])}", metadata=q)
        for q in question_bank
    ]
    
    vector_store = FAISS.from_documents(documents, embeddings)
    print("✅ Done.")

    interview_package = []
    all_skills = parsed_jd.required_skills + parsed_jd.preferred_skills
    
    # --- Find Matching Questions from Bank ---
    print("\n[3/4] Finding relevant questions from the question bank...")
    retriever = vector_store.as_retriever(search_kwargs={"k": 1}) # Get the top 1 match for each skill
    
    # To avoid duplicates
    used_question_ids = set()

    for skill in set(all_skills): # Use set to avoid searching for the same skill twice
        # Find the most similar question in our bank for the current skill
        retrieved_docs = retriever.invoke(skill)
        if retrieved_docs:
            matched_question = retrieved_docs[0].metadata
            if matched_question['id'] not in used_question_ids:
                print(f"  -> Found match for '{skill}': {matched_question['question'][:40]}...")
                interview_package.append(matched_question)
                used_question_ids.add(matched_question['id'])

    print(f"✅ Done. Found {len(interview_package)} questions from the bank.")

    # --- Generate New, Tailored Questions ---
    print("\n[4/4] Generating new, tailored questions...")
    generator_chain = get_generator_chain()
    
    # We will generate 2 new questions based on the most important required skills.
    skills_for_generation = parsed_jd.required_skills[:2] 
    for skill in skills_for_generation:
        print(f"  -> Generating question for '{skill}'...")
        new_question = generator_chain.invoke({
            "job_title": parsed_jd.job_title,
            "skill": skill,
            "experience": parsed_jd.years_of_experience
        })
        interview_package.append(new_question.dict())

    print("✅ Done.")
    
    # --- Final Output ---
    print("\n\n🎉 === Your Custom Interview Package === 🎉")
    print(json.dumps(interview_package, indent=2))
    print("\n==========================================")


if __name__ == "__main__":
    main()