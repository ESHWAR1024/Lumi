"""
Gemini API service for generating emotion-based picture prompts.
Supports multiple API keys for load balancing.
"""

import os
import google.generativeai as genai
from typing import List, Dict
import json
import random

class GeminiPromptsService:
    def __init__(self, api_keys: List[str]):
        """
        Initialize Gemini service with multiple API keys.
        
        Args:
            api_keys: List of Gemini API keys for load balancing
        """
        self.api_keys = api_keys
        self.current_key_index = 0
        self.model_name = "gemini-2.5-flash"  # Gemini 2.5 Flash (latest, fastest, free tier)
        
        if not self.api_keys:
            raise ValueError("At least one Gemini API key is required")
        
        # Configure with first key
        self._configure_api()
    
    def _configure_api(self):
        """Configure Gemini API with current key."""
        genai.configure(api_key=self.api_keys[self.current_key_index])
        self.model = genai.GenerativeModel(self.model_name)
    
    def _rotate_key(self):
        """Rotate to next API key."""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self._configure_api()
        print(f"🔄 Rotated to API key {self.current_key_index + 1}/{len(self.api_keys)}")
    
    def generate_initial_prompts(self, emotion: str) -> List[Dict[str, str]]:
        """
        Generate 4 initial picture prompts based on detected emotion.
        
        Args:
            emotion: Detected emotion (happy, sad, angry, etc.)
        
        Returns:
            List of 4 prompt dictionaries with 'label' and 'description'
        """
        prompt = f"""You are helping a child with neurological disabilities communicate their emotions.
The child is feeling {emotion}.

Generate exactly 4 simple, clear reasons why a child might feel {emotion}. 
Each reason should be:
- Simple and easy to understand (suitable for children)
- Represented by a short label (2-4 words)
- Accompanied by a brief description

Format your response as a JSON array with exactly 4 objects, each having:
- "label": A short, simple label (2-4 words)
- "description": A brief description (one sentence)

Example format:
[
  {{"label": "Friends and Family", "description": "Something about friends or family members"}},
  {{"label": "School or Learning", "description": "Something about school or learning activities"}},
  {{"label": "Food or Eating", "description": "Something about food or mealtime"}},
  {{"label": "Play or Activities", "description": "Something about playing or activities"}}
]

Generate 4 appropriate reasons for feeling {emotion}:"""

        try:
            response = self.model.generate_content(prompt)
            
            # Parse JSON response
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            prompts = json.loads(response_text)
            
            # Validate we have exactly 4 prompts
            if len(prompts) != 4:
                raise ValueError(f"Expected 4 prompts, got {len(prompts)}")
            
            return prompts
            
        except Exception as e:
            print(f"❌ Error generating initial prompts: {e}")
            # Rotate key and retry once
            self._rotate_key()
            try:
                response = self.model.generate_content(prompt)
                response_text = response.text.strip()
                
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                
                response_text = response_text.strip()
                prompts = json.loads(response_text)
                
                if len(prompts) != 4:
                    raise ValueError(f"Expected 4 prompts, got {len(prompts)}")
                
                return prompts
            except Exception as retry_error:
                print(f"❌ Retry failed: {retry_error}")
                # Return fallback prompts
                return self._get_fallback_initial_prompts(emotion)
    
    def generate_followup_prompts(self, emotion: str, selected_category: str) -> List[Dict[str, str]]:
        """
        Generate 4 follow-up picture prompts based on emotion and selected category.
        
        Args:
            emotion: Detected emotion
            selected_category: The category the child selected
        
        Returns:
            List of 4 specific prompt dictionaries
        """
        prompt = f"""You are helping a child with neurological disabilities communicate their emotions.
The child is feeling {emotion} and has indicated it's related to "{selected_category}".

Generate exactly 4 specific, simple reasons related to "{selected_category}" that might make a child feel {emotion}.
Each reason should be:
- Very specific and actionable
- Simple and easy to understand
- Represented by a short label (3-5 words)
- Accompanied by a brief description

Format your response as a JSON array with exactly 4 objects, each having:
- "label": A short, specific label (3-5 words)
- "description": A brief description (one sentence)

Example format for "Friends and Family" + "sad":
[
  {{"label": "Had a fight with sibling", "description": "Got into an argument with brother or sister"}},
  {{"label": "Missing a family member", "description": "Want to see someone who is away"}},
  {{"label": "Friend is upset with me", "description": "A friend is angry or not talking to me"}},
  {{"label": "Left out by friends", "description": "Friends are playing without me"}}
]

Generate 4 specific reasons for feeling {emotion} related to "{selected_category}":"""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            prompts = json.loads(response_text)
            
            if len(prompts) != 4:
                raise ValueError(f"Expected 4 prompts, got {len(prompts)}")
            
            return prompts
            
        except Exception as e:
            print(f"❌ Error generating followup prompts: {e}")
            self._rotate_key()
            try:
                response = self.model.generate_content(prompt)
                response_text = response.text.strip()
                
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                
                response_text = response_text.strip()
                prompts = json.loads(response_text)
                
                if len(prompts) != 4:
                    raise ValueError(f"Expected 4 prompts, got {len(prompts)}")
                
                return prompts
            except Exception as retry_error:
                print(f"❌ Retry failed: {retry_error}")
                return self._get_fallback_followup_prompts(emotion, selected_category)
    
    def _get_fallback_initial_prompts(self, emotion: str) -> List[Dict[str, str]]:
        """Fallback prompts if API fails."""
        fallbacks = {
            "sad": [
                {"label": "Friends and Family", "description": "Something about friends or family members"},
                {"label": "School or Learning", "description": "Something about school or learning"},
                {"label": "Food or Eating", "description": "Something about food or mealtime"},
                {"label": "Play or Activities", "description": "Something about playing or activities"}
            ],
            "happy": [
                {"label": "Friends and Family", "description": "Something good with friends or family"},
                {"label": "School or Learning", "description": "Something good at school"},
                {"label": "Food or Eating", "description": "Something good about food"},
                {"label": "Play or Activities", "description": "Something fun to do"}
            ],
            "angry": [
                {"label": "Friends and Family", "description": "Something upsetting with friends or family"},
                {"label": "School or Learning", "description": "Something frustrating at school"},
                {"label": "Rules or Limits", "description": "Something about rules or being told no"},
                {"label": "Things Not Working", "description": "Something not working as expected"}
            ]
        }
        
        return fallbacks.get(emotion.lower(), fallbacks["sad"])
    
    def _get_fallback_followup_prompts(self, emotion: str, category: str) -> List[Dict[str, str]]:
        """Fallback follow-up prompts if API fails."""
        return [
            {"label": f"Specific issue 1", "description": f"A specific situation related to {category}"},
            {"label": f"Specific issue 2", "description": f"Another situation related to {category}"},
            {"label": f"Specific issue 3", "description": f"A third situation related to {category}"},
            {"label": f"Specific issue 4", "description": f"A fourth situation related to {category}"}
        ]
