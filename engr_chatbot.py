import vertexai
from vertexai.generative_models import GenerativeModel, Part
import os

# --- CONFIGURATION ---
PROJECT_ID = "speedy-cab-463315-v3"
REGION = "us-central1"

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=REGION)

# Load Gemini model
model = GenerativeModel("gemini-1.5-flash")

# --- Load Engineering Best Practices Document ---
try:
    with open("engineering_best_practices.txt", "r", encoding="utf-8") as f:
        engineering_best_practices = f.read()
except FileNotFoundError:
    print("Error: 'engineering_best_practices.txt' not found.")
    exit()
    
def get_engineering_assistance(query: str) -> str:
    """
    Generates answers to engineering-related queries using the Gemini model.
    """
    # Dynamic Prompting: Adjusting prompt based on type of query
    # This is a simple heuristic; more complex bots would use NLU to categorize intent. 
    
    base_instruction = """
    You are an internal AI assistant for a software engineering team. Your goal is to provide helpful, concise, and accurate responses related to coding,
    code explanation, and adherence to engineering best practices.
    Always prioritize the best practices. 
    If a query asks for a code snippet, provide it in a code block.
    If you cannot fully answer based on your knowledge and the provided best practices, state that and suggest contacting a senior engineering or looking
    up official documentation.
    """
    
    # Best practiecs as grounding context
    grounding_context = f"""
    ## Engineering Best Practices:
    {engineering_best_practices}
    """
    
    # Craft full prompt
    prompt = f"""
    {base_instruction}
    
    {grounding_context}
    
    ## Engineer's Query: 
    {query}
    
    ## Your Response:
    """
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 1000,
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 40,
            }
        )
        return response.text.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"


def main():
    """
    Main function to run the engineering chatbot in a loop.
    """
    print("Welcome to the Engineering Chatbot!")
    print("I can help with coding questions, code explanations, and engineering best practices.")
    print("Type 'exit' or 'quit' to quit the chatbot.\n")
    
    while True:
        user_query = input("\nEngineer's question: ")
        
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting the chatbot. Goodbye!")
            break
        
        if not user_query.strip():
            print("Please enter a question.")
            continue
        
        print("Analyzing query and generating response...")
        answer = get_engineering_assistance(user_query)
        print(f"Assistant: {answer}")
        
if __name__ == "__main__":
    main()