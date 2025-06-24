import vertexai
from vertexai.generative_models import GenerativeModel, Part
from vertexai.preview.generative_models import GenerationConfig
import os
import tiktoken

# --- CONFIGURATION ---
PROJECT_ID = "speedy-cab-463315-v3"
REGION = "us-central1"

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=REGION)

# Load Gemini model
model = GenerativeModel("gemini-2.5-pro")

# --- Load Engineering Best Practices Document ---
try:
    with open("engineering_best_practices.txt", "r", encoding="utf-8") as f:
        engineering_best_practices = f.read()
except FileNotFoundError:
    print("Error: 'engineering_best_practices.txt' not found.")
    exit()
    
def estimate_tokens(text: str) -> int:
    """
    Rough token estimation
    """
    return len(text) // 4

def trim_context(context: str, max_tokens: int = 2000) -> str:
    """
    Trims context to fit within token limits.
    """
    tokens = estimate_tokens(context)
    if estimate_tokens(context) > max_tokens:
        return context
    
    char_limit = max_tokens * 4  # Roughly 4 characters per token
    trimmed = context[:char_limit]
    return trimmed + "\n\n(...content truncated to fit token limit)"

def get_engineering_assistance(query: str) -> str:
    """
    Generates answers to engineering-related queries using the Gemini model.
    """
    # Dynamic Prompting: Adjusting prompt based on type of query
    # This is a simple heuristic; more complex bots would use NLU to categorize intent. 
    
    base_instruction = (
        "You are an internal AI assistant for a software engineering team. "
        "Provide concise, accurate answers about code, explanations, and engineering best practices. "
        "Use code blocks when sharing code. If unsure, recommend consulting a senior engineer or documentation."
    )
    
    # Best practiecs as grounding context
    grounding_context = ""
    if engineering_best_practices:
        if any(keyword in query.lower() for keyword in ['write','code','create','build','make',]):
            trimmed_practices = trim_context(engineering_best_practices, max_tokens=800)
        else:
            trimmed_practices = trim_context(engineering_best_practices, max_tokens=1500)
        grounding_context = f"## Engineering Best Practices:\n{trimmed_practices}\n\n"
        
    
    full_prompt = f"{base_instruction}\n\n{grounding_context}## Engineer's Query:\n{query}\n\n## Your Response:"

    # Check token limits
    prompt_tokens = estimate_tokens(full_prompt)
    print(f"Estimated tokens in prompt: {prompt_tokens}")    
    try:
        response = model.generate_content(
            full_prompt,
            generation_config = GenerationConfig(
                max_output_tokens=3000,
                temperature=0.3,
                top_p=0.9,
                top_k=40
            )
        )
        # Better error handling
        if not response.candidates:
            return "Error: No response candidates generated."
        
        candidate = response.candidates[0]
        
        # Check finish reason
        finish_reason = candidate.finish_reason
        
        # Map numeric finish reasons to name
        finish_reason_map = {
            1: "STOP",
            2: "MAX_TOKENS",
            3: "SAFETY",
            4: "RECITATION",
            5: "OTHER"
        }
        
        if isinstance(finish_reason, int):
            finish_reason_name = finish_reason_map.get(finish_reason, f"UNKNOWN_{finish_reason}")
        else:
            finish_reason_name = str(finish_reason)
            
        print(f"Finish reason: {finish_reason_name}")
        
        if finish_reason_name == "MAX_TOKENS":
            return "Response was truncated due to length limits. Please ask a more specific question."
        elif finish_reason_name == "SAFETY":
            return "Response was blocked by safety filters. Please rephrase your question."
        elif finish_reason_name == "RECITATION":
            return "Response was blocked due to recitation concerns. Please rephrase your question."
        elif finish_reason_name in ["STOP", "FINISH_REASON_UNSPECIFIED"]:
            # Normal completion - try to get the text
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                return candidate.content.parts[0].text.strip()
            elif hasattr(candidate, 'text'):
                return candidate.text.strip()
            else:
                return "Error: Response content is empty despite normal completion."
        else:
            # For any other finish reason, still try to get text if available
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                text = candidate.content.parts[0].text.strip()
                if text:
                    return text
            return f"Unexpected finish reason: {finish_reason_name}. No response text available."
        
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
        try:
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
        
        except KeyboardInterrupt:
            print("\nExiting the chatbot. Goodbye!")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            continue
        
if __name__ == "__main__":
    main()