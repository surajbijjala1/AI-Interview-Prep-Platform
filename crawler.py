import requests
import json
from bs4 import BeautifulSoup

def crawl_questions():
    """
    Crawls a specific webpage for interview questions and saves them to a JSON file.
    NOTE: This is a fragile script. If the target website's HTML structure changes, this will break.
    """
    # We will use a reliable, static example page for this demonstration.
    # This page is designed to be scraped easily.
    url = "https://gist.githubusercontent.com/surajbijjala1/c72a5d8d31b81bdb91dc1a6740492483/raw/d4a43f5f07c09c017aa461b3d32bfa9cbaf3c9be/myquestions.html"
    print("🚀 Starting crawler...")
    print(f"Fetching questions from: {url}")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Raises an exception for bad status codes (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching the URL: {e}")
        return

    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')

    scraped_questions = []
    
    # The target HTML has questions in 'div' tags with the class 'question-block'
    question_blocks = soup.find_all('div', class_='question-block')
    
    if not question_blocks:
        print("⚠️ No questions found. The website structure might have changed.")
        return

    print(f"Found {len(question_blocks)} questions. Parsing them now...")

    for block in question_blocks:
        try:
            # Extract data based on the HTML structure
            question_text = block.find('h3', class_='question').text.strip()
            q_type = block.find('span', class_='type').text.strip()
            difficulty = block.find('span', class_='difficulty').text.strip()
            # The 'id' and 'tags' are stored in data attributes
            q_id = block['data-id']
            tags = block['data-tags'].split(',')

            scraped_questions.append({
                "id": q_id,
                "type": q_type,
                "difficulty": difficulty,
                "question": question_text,
                "tags": [tag.strip() for tag in tags] # Clean up whitespace
            })
        except (AttributeError, KeyError) as e:
            print(f"Skipping a block due to missing data: {e}")
            continue

    # Save the results to a file
    output_filename = "questions_crawled.json"
    with open(output_filename, 'w') as f:
        json.dump(scraped_questions, f, indent=2)

    print(f"\n✅ Success! Scraped {len(scraped_questions)} questions and saved them to `{output_filename}`.")

if __name__ == "__main__":
    crawl_questions()
