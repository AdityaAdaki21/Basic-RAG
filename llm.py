# llm.py

import requests
import json
import time
import logging
from typing import Dict, Any, Optional, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMClient:
    """Client for interacting with Ollama LLM API."""
    
    def __init__(self, base_url="http://localhost:11434", default_model="llama3.2"):
        self.base_url = base_url.rstrip('/')
        self.default_model = default_model
        self.generate_endpoint = f"{self.base_url}/api/generate"
        self.model_endpoint = f"{self.base_url}/api/show"
        self.loaded_model = None
        self.ensure_model_loaded(default_model)  # Load the model when client is initialized
    
    def ensure_model_loaded(self, model):
        """Ensure the specified model is loaded and ready for use."""
        if model == self.loaded_model:
            return  # Model is already loaded
        
        try:
            # Check if the model is loaded
            response = requests.post(
                self.model_endpoint,
                json={"name": model},
                timeout=(5, 10)
            )
            
            if response.status_code == 200:
                self.loaded_model = model
                logger.info(f"Model {model} is already loaded")
            else:
                # If not loaded, load it
                logger.info(f"Loading model {model}...")
                response = requests.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model},
                    timeout=(5, 300)  # Longer timeout for model loading
                )
                response.raise_for_status()
                self.loaded_model = model
                logger.info(f"Model {model} successfully loaded")
        except Exception as e:
            logger.warning(f"Failed to ensure model is loaded: {e}")


    def generate(self, 
                prompt: str, 
                model: Optional[str] = None,
                temperature: float = 0.7,
                max_tokens: Optional[int] = None,
                stream: bool = False,
                max_retries: int = 3,
                retry_delay: int = 1) -> Union[str, Dict[str, Any]]:
        """
        Generate text using the LLM with retry logic.
        
        Args:
            prompt: The input prompt for the model
            model: Model to use (defaults to instance default_model)
            temperature: Controls randomness (0.0-1.0)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            Generated text or full response object if raw_response=True
        """
        model = model or self.default_model
        
        # Build the request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        # Add optional parameters if provided
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        # Implement retry logic
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.generate_endpoint,
                    json=payload,
                    timeout=(5, 120)  # (connect_timeout, read_timeout)
                )
                
                response.raise_for_status()  # Raise exception for HTTP errors
                
                result = response.json()
                if "response" in result:
                    return result["response"]
                else:
                    raise ValueError(f"Unexpected response format: {result}")
                    
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"API request failed (attempt {attempt+1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to generate text after {max_retries} attempts: {e}")
                    return f"Error: Failed to generate response after {max_retries} attempts. Please check the Ollama server and try again."
    
    def generate_with_context(self, 
                          query: str,
                          context: str, 
                          model: Optional[str] = None,
                          temperature: float = 0.7,
                          max_tokens: Optional[int] = None,
                          stream: bool = False,
                          max_retries: int = 3,
                          retry_delay: int = 1) -> Union[str, Dict[str, Any]]:
        """
        Generate a response based on a query and context information.
        
        Args:
            query: The user's query
            context: Context information to inform the response
            model: Model to use
            
        Returns:
            Generated response
        """
        model = model or self.default_model
        # Ensure the model is loaded
        if model != self.loaded_model:
            self.ensure_model_loaded(model)

        prompt = f"""Context information is below.
    ---------------------
    {context}
    ---------------------
    Given the context information and not prior knowledge, answer the question: {query}

    If the context doesn't contain relevant information to answer the question,
    respond with "I don't have enough information to answer that question."
    """

        return self.generate(prompt, model=model)


# For backward compatibility
def call_ollama(prompt, model="llama3.2"):
    """Legacy function for backward compatibility."""
    client = LLMClient()
    return client.generate(prompt, model=model)