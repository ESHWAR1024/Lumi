"""
Enhanced Gemini API service with context-aware prompt generation.
Uses child profile, routine, and time context for intelligent responses.
"""

import google.generativeai as genai
from typing import List, Dict
import json
from datetime import datetime

class EnhancedGeminiService:
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.model_name = "gemini-2.5-flash"
        
        if not self.api_keys:
            raise ValueError("At least one Gemini API key is required")
        
        self._configure_api()
    
    def _configure_api(self):
        genai.configure(api_key=self.api_keys[self.current_key_index])
        self.model = genai.GenerativeModel(self.model_name)
    
    def _rotate_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self._configure_api()
        print(f"🔄 Rotated to API key {self.current_key_index + 1}/{len(self.api_keys)}")
    
    def generate_initial_prompts(
        self, 
        emotion: str, 
        child_name: str,
        age: int,
        diagnosis: str,
        routine: Dict,
        current_time: str
    ) -> List[Dict[str, str]]:
        """
        Generate context-aware initial prompts using 90% AI intelligence + 10% database context.
        """
        
        prompt = f"""You are an empathetic AI assistant helping a child with neurological disabilities communicate their emotions.

CHILD CONTEXT:
- Name: {child_name}
- Age: {age} years old
- Diagnosis: {diagnosis}
- Current Time: {current_time}

DAILY ROUTINE:
- Wake up: {routine.get('wake_up_time', 'Not specified')}
- Breakfast: {routine.get('breakfast_time', 'Not specified')}
- Lunch: {routine.get('lunch_time', 'Not specified')}
- Snacks: {routine.get('snacks_time', 'Not specified')}
- Dinner: {routine.get('dinner_time', 'Not specified')}
- Bedtime: {routine.get('bedtime', 'Not specified')}

PREFERENCES:
- Favorite Activities: {routine.get('favorite_activities', 'Not specified')}
- Comfort Items: {routine.get('comfort_items', 'Not specified')}
- Preferred Communication: {routine.get('preferred_prompts', 'Not specified')}

CURRENT SITUATION:
{child_name} is feeling {emotion} right now at {current_time}.

TASK:
Generate exactly 4 picture prompts that could explain why {child_name} is feeling {emotion}.

IMPORTANT GUIDELINES:
1. Use 85-90% of your AI intelligence to understand the context
2. Consider the child's age, diagnosis, and current time
3. Reference their routine (e.g., if it's near snack time and they're sad, suggest hunger)
4. Consider their favorite activities and comfort items
5. Think about common challenges for children with {diagnosis}
6. Make prompts simple, clear, and age-appropriate for a {age}-year-old
7. Use the child's name ({child_name}) to make it personal

FORMAT:
Return ONLY a JSON array with exactly 4 objects:
[
  {{
    "label": "Short label (2-4 words)",
    "description": "Brief, child-friendly description (one sentence)",
    "reasoning": "Why this is relevant given the context"
  }}
]

Generate the 4 most relevant prompts now:"""

        try:
            response = self.model.generate_content(prompt)
            response_text = self._clean_json_response(response.text)
            prompts = json.loads(response_text)
            
            if len(prompts) != 4:
                raise ValueError(f"Expected 4 prompts, got {len(prompts)}")
            
            return prompts
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self._rotate_key()
            try:
                response = self.model.generate_content(prompt)
                response_text = self._clean_json_response(response.text)
                prompts = json.loads(response_text)
                return prompts
            except:
                return self._get_fallback_prompts(emotion, child_name, age)
    
    def generate_followup_prompts(
        self,
        emotion: str,
        selected_option: str,
        child_name: str,
        age: int,
        diagnosis: str,
        routine: Dict,
        current_time: str,
        conversation_history: List[str]
    ) -> List[Dict[str, str]]:
        """
        Generate intelligent follow-up prompts to dig deeper into the reason.
        """
        
        history_text = " → ".join(conversation_history + [selected_option])
        
        prompt = f"""You are helping {child_name} ({age} years old, {diagnosis}) communicate why they're feeling {emotion}.

CONVERSATION SO FAR:
{history_text}

CONTEXT:
- Current Time: {current_time}
- Diagnosis: {diagnosis}
- Favorite Activities: {routine.get('favorite_activities', 'Not specified')}
- Comfort Items: {routine.get('comfort_items', 'Not specified')}

{child_name} just selected "{selected_option}" as a reason for feeling {emotion}.

TASK:
Use your intelligence to generate 4 specific follow-up prompts that dig deeper into WHY they selected "{selected_option}".

GUIDELINES:
1. Be very specific and actionable
2. Consider the child's diagnosis and how it affects them
3. Think about what a {age}-year-old with {diagnosis} might experience
4. Reference their routine and preferences when relevant
5. Make it easy for the child to identify the exact problem

FORMAT:
Return ONLY a JSON array with exactly 4 objects:
[
  {{
    "label": "Specific reason (3-5 words)",
    "description": "Clear, simple description",
    "reasoning": "Why this makes sense given {selected_option}"
  }}
]

Generate the 4 most relevant follow-up prompts:"""

        try:
            response = self.model.generate_content(prompt)
            response_text = self._clean_json_response(response.text)
            prompts = json.loads(response_text)
            
            if len(prompts) != 4:
                raise ValueError(f"Expected 4 prompts, got {len(prompts)}")
            
            return prompts
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return self._get_fallback_followup(selected_option)
    
    def generate_solution(
        self,
        emotion: str,
        conversation_history: List[str],
        child_name: str,
        age: int,
        diagnosis: str,
        routine: Dict
    ) -> str:
        """
        Generate an empathetic, actionable solution based on the conversation.
        """
        
        history_text = " → ".join(conversation_history)
        
        prompt = f"""You are providing a solution for {child_name} ({age} years old, {diagnosis}).

PROBLEM IDENTIFIED:
{child_name} is feeling {emotion} because: {history_text}

CONTEXT:
- Diagnosis: {diagnosis}
- Favorite Activities: {routine.get('favorite_activities', 'Not specified')}
- Comfort Items: {routine.get('comfort_items', 'Not specified')}

TASK:
Generate a warm, empathetic, and actionable solution that:
1. Acknowledges {child_name}'s feelings
2. Provides 2-3 specific, simple actions they or their caregiver can take
3. Is appropriate for a {age}-year-old with {diagnosis}
4. References their comfort items or favorite activities if relevant
5. Is encouraging and supportive

Keep it under 100 words and use simple language.

Generate the solution now:"""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"❌ Error: {e}")
            return f"I understand you're feeling {emotion}, {child_name}. Let's work together to make things better. Would you like to try one of your favorite activities or get your comfort item?"
    
    def _clean_json_response(self, text: str) -> str:
        """Remove markdown code blocks from response."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
    
    def _get_fallback_prompts(self, emotion: str, child_name: str, age: int) -> List[Dict[str, str]]:
        """Fallback prompts if API fails."""
        return [
            {"label": "Need Something", "description": f"{child_name} might need something", "reasoning": "Basic need"},
            {"label": "Feeling Tired", "description": f"{child_name} might be tired", "reasoning": "Common for children"},
            {"label": "Want Activity", "description": f"{child_name} wants to do something", "reasoning": "Activity desire"},
            {"label": "Need Help", "description": f"{child_name} needs help with something", "reasoning": "Assistance needed"}
        ]
    
    def _get_fallback_followup(self, selected: str) -> List[Dict[str, str]]:
        """Fallback follow-up prompts."""
        return [
            {"label": "Option 1", "description": f"Related to {selected}", "reasoning": "Follow-up"},
            {"label": "Option 2", "description": f"Another aspect of {selected}", "reasoning": "Follow-up"},
            {"label": "Option 3", "description": f"Different angle on {selected}", "reasoning": "Follow-up"},
            {"label": "Option 4", "description": f"Alternative for {selected}", "reasoning": "Follow-up"}
        ]
