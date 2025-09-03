import os
import json
import time
from dotenv import load_dotenv
from typing import List, Dict


# --- LangChain Imports ---
# Note: The deprecation warnings are from LangChain and can be ignored for now.
# They are updating their library structure.
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.schema.document import Document
from langchain.vectorstores.faiss import FAISS

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
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0, convert_system_message_to_human=True)
    prompt = ChatPromptTemplate.from_template("From the job description, extract information into the provided schema.\n\nJob Description:\n{job_description}")
    return prompt | llm.with_structured_output(ParsedJD)

def get_generator_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.5, convert_system_message_to_human=True)
    prompt = ChatPromptTemplate.from_template("You are an expert interviewer. Generate one high-quality question based on the following details:\nJob Title: {job_title}\nRequired Skill: {skill}\nYears of Experience: {experience}\nReturn the question with all required metadata.")
    return prompt | llm.with_structured_output(GeneratedQuestion)

def get_skill_graph_chain():
    """Returns a chain that categorizes a list of skills using the corrected model."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0, convert_system_message_to_human=True)
    prompt = ChatPromptTemplate.from_template(
        "You are a tech skills taxonomist. Categorize the following list of skills into a list of logical groups. "
        "Each group should have a category name and a list of the corresponding skills from the input list. "
        "For example: 'Programming Language', 'Web Framework', 'Database', 'Cloud Platform', 'DevOps Tools', etc. "
        "\n\nSkills:\n{skill_list}"
    )
    return prompt | llm.with_structured_output(SkillGraph)

# --- 3. MAIN ORCHESTRATION FUNCTION ---
def main():
    print("🚀 Starting Full-Featured Interview Generator...")

    question_file = "questions_crawled.json"
    try:
        with open(question_file, "r") as f:
            question_bank = json.load(f)
        print(f"Successfully loaded {len(question_bank)} questions from `{question_file}`.")
    except FileNotFoundError:
        print(f"❌ Error: `{question_file}` not found. Please run `python crawler.py` first.")
        return

    jd_text = """
    We create AI solutions to help people navigate and rise above barriers. 



Role Description

This is a 6-month unpaid Software Engineer internship opportunity at High Tide. The Software Engineer Intern will be responsible for assisting in the development and maintenance of full-stack AI applications. Daily tasks will include debugging and testing software, collaborating with team members to optimize code, designing core systems, and participating in code reviews.


Qualifications

Experience in Software Development and AI (LLMs, Agentic AI, Prompt Engineering)
Proficiency in Object-Oriented Programming (OOP) and DSA
Strong problem-solving skills and ability to learn new technologies quickly
Excellent written and verbal communication skills
Ability to work independently and as part of a team
Currently pursuing or recently completed a degree in Computer Science or Data Science; or recent CS/AI Bootcamp Grad
    """
    
    print("\n[1/7] Parsing Job Description...")
    parsed_jd = get_parser_chain().invoke({"job_description": jd_text})
    print(f"✅ Done. Role: {parsed_jd.job_title}, Experience: {parsed_jd.years_of_experience} years.")

    print("\n[2/7] Building Role-Specific Skill Graph...")
    all_skills = parsed_jd.required_skills + parsed_jd.preferred_skills
    skill_graph_result = get_skill_graph_chain().invoke({"skill_list": ", ".join(all_skills)})
    print("✅ Done. Skill Ontology Generated:")
    # --- CORRECTED PRINTING LOGIC ---
    for category_obj in skill_graph_result.graph:
        print(f"  - {category_obj.category_name}:")
        for skill in category_obj.skills:
            print(f"    - {skill}")
    
    print("\n[3/7] Creating Vector Store for question matching...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    documents = [Document(page_content=f"Question: {q['question']} Tags: {', '.join(q['tags'])}", metadata=q) for q in question_bank]
    vector_store = FAISS.from_documents(documents, embeddings)
    print("✅ Done.")

    interview_package = []
    used_question_ids = set()

    print("\n[4/7] Finding relevant questions from the question bank...")
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    for skill in set(all_skills):
        retrieved_docs = retriever.invoke(skill)
        if retrieved_docs:
            matched_question = retrieved_docs[0].metadata
            if matched_question['id'] not in used_question_ids:
                interview_package.append(matched_question)
                used_question_ids.add(matched_question['id'])
    print(f"✅ Done. Found {len(interview_package)} relevant questions.")
    
    print("\n[5/7] Generating new, tailored questions...")
    skills_for_generation = parsed_jd.required_skills[:2]
    for skill in skills_for_generation:
        print(f"  -> Generating question for '{skill}'...")
        new_question = get_generator_chain().invoke({"job_title": parsed_jd.job_title, "skill": skill, "experience": parsed_jd.years_of_experience})
        interview_package.append(new_question.dict())
    print("✅ Done.")

    print("\n[6/7] Finalizing interview package...")
    time.sleep(1)

    print("\n" + "="*50)
    print("🎉 Interview Package Generated. Starting Interactive Demo...")
    print("="*50 + "\n")
    time.sleep(2)

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
