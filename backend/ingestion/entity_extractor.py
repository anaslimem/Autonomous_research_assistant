import google.generativeai as genai
import json
import os 
from dotenv import load_dotenv
import logging
import re

# Set up logging
logger = logging.getLogger(__name__)

load_dotenv()  # Load environment variables from .env file

GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL = os.getenv("MODEL")

if not GENAI_API_KEY:
    raise EnvironmentError("Missing GENAI_API_KEY environment variable")

genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel(MODEL)

def extract_entities(text: str, title: str = None) -> dict:
    """
    Extract entities from text using Gemini.
    
    Returns:
        {
            "authors": ["name1", "name2"],
            "topics": ["topic1", "topic2"],
            "technologies": ["tech1", "tech2"],
            "companies": ["company1", "company2"],
            "concepts": ["concept1", "concept2"]
        }
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for entity extraction")
        return {"authors": [], "topics": [], "technologies": [], "companies": [], "concepts": []}
    
    prompt = f"""
Extract entities from this article. Return ONLY valid JSON, no markdown code blocks.

Article Title: {title or "Unknown"}
Article Text (first 3000 chars): {text[:3000]}

Extract:
- authors: People who wrote or are mentioned as authors
- topics: Main subjects/themes (e.g., "machine learning", "NLP")
- technologies: Tools, frameworks, models (e.g., "PyTorch", "GPT-4", "BERT")
- companies: Organizations mentioned (e.g., "OpenAI", "Google")
- concepts: Key technical concepts (e.g., "attention mechanism", "fine-tuning")

Return JSON format:
{{"authors": [], "topics": [], "technologies": [], "companies": [], "concepts": []}}
"""
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = re.sub(r"```json?\n?", "", response_text)
            response_text = re.sub(r"```\n?$", "", response_text)
        
        entities = json.loads(response_text)
        logger.info(f"Extracted entities: {len(entities.get('topics', []))} topics, {len(entities.get('technologies', []))} technologies")
        return entities
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini response as JSON: {e}")
        return {"authors": [], "topics": [], "technologies": [], "companies": [], "concepts": []}
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        return {"authors": [], "topics": [], "technologies": [], "companies": [], "concepts": []}
