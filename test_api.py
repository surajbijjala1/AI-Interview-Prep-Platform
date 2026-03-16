import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
print(f"API Key loaded: {api_key[:20]}..." if api_key else "No API key found!")

if api_key:
    print("\n🧪 Testing with LangChain's ChatGoogleGenerativeAI...")
    
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    # Test different model names
    models_to_test = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-1.0-pro",
        "models/gemini-1.5-flash",
    ]
    
    for model_name in models_to_test:
        try:
            print(f"\n  Testing: {model_name}...", end=" ")
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
            response = llm.invoke("Say 'works'")
            print(f"✅ SUCCESS! Response: {response.content[:50]}")
            print(f"\n🎉 USE THIS MODEL: {model_name}")
            break  # Stop at first working model
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "not found" in error_msg.lower():
                print(f"❌ Not found")
            elif "PERMISSION_DENIED" in error_msg:
                print(f"❌ Permission denied")
            elif "API key not valid" in error_msg:
                print(f"❌ Invalid API key")
            else:
                print(f"❌ Error: {error_msg[:100]}")
