# llm.py

import requests
import json

def call_ollama(prompt, model="llama3.2"):
    """
    Call the Ollama API to generate text using the specified model and prompt.
    Returns the generated text.
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        
        if response.status_code != 200:
            print(f"Ollama API error: {response.text}")
            return f"Error: Ollama API returned status code {response.status_code}"
            
        result = response.json()
        if "response" in result:
            return result["response"]
        else:
            return f"Error: Unexpected response format: {result}"
            
    except Exception as e:
        return f"Error calling Ollama API: {e}"