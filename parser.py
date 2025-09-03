
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional

# Load the API key from the .env file
load_dotenv()

# 1. Define the data structure we want to extract (our schema)
# This uses Pydantic to ensure the output is structured correctly.
class ParsedJD(BaseModel):
    job_title: str = Field(description="The official title of the job role.")
    required_skills: List[str] = Field(description="A list of essential skills, tools, or technologies required for the role.")
    preferred_skills: List[str] = Field(description="A list of skills that are preferred but not mandatory.")
    years_of_experience: int = Field(description="The minimum number of years of professional experience required.")
    key_responsibilities: List[str] = Field(description="A list of the primary responsibilities for this role.")

# 2. Define our LLM
# We use GPT-4o for its strong reasoning and JSON output capabilities.
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0, convert_system_message_to_human=True)

# 3. Create the prompt template
prompt_template = """
From the following job description, extract the required information and format it according to the provided schema.
Focus only on the details present in the text.

Job Description:
---
{job_description}
---
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# 4. Create the "chain" that links the prompt, LLM, and output schema
# .with_structured_output() is a powerful feature that forces the LLM to return JSON matching our ParsedJD model.
parser_chain = prompt | llm.with_structured_output(ParsedJD)

# 5. Provide a sample Job Description to test
sample_jd = """
Senior Python Developer

Company: Tech Solutions Inc.
Location: Remote

We are seeking a Senior Python Developer with at least 5 years of experience to join our dynamic team. The ideal candidate will be responsible for developing and maintaining our core backend services.

Responsibilities:
- Design and implement scalable backend systems.
- Write clean, maintainable, and efficient code.
- Collaborate with front-end developers to integrate user-facing elements.
- Troubleshoot and debug applications.

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

# 6. Run the chain and print the result
if __name__ == "__main__":
    print("Parsing Job Description...")
    parsed_result = parser_chain.invoke({"job_description": sample_jd})
    
    # Print the result in a readable format
    import json
    print(json.dumps(parsed_result.dict(), indent=2))