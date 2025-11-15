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
    
    def get_next_api_key(self):
        """Get the next API key in rotation"""
        key = self.api_keys[self.current_key_index]
        self._rotate_key()
        return key
    
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
        Enhanced with condition-specific knowledge.
        """
        
        # Import condition profiles
        from condition_profiles import get_condition_specific_prompt_guidance
        
        # Get condition-specific guidance
        condition_guidance = get_condition_specific_prompt_guidance(diagnosis, emotion, age)
        
        prompt = f"""You are an AI assistant with the expertise of a pediatric behavioral therapist, specializing in helping children with neurological disabilities communicate their emotions.

THINK SYSTEMATICALLY LIKE A THERAPIST:
1. Consider the child's developmental stage and diagnosis
2. Identify root causes, not just symptoms
3. Think about physical, emotional, and environmental factors
4. Prioritize actionable, evidence-based solutions
5. Communicate warmly and age-appropriately

CHILD CONTEXT:
- Name: {child_name}
- Age: {age} years old
- Diagnosis: {diagnosis}
- Current Time: {current_time}

{condition_guidance}

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

THERAPEUTIC ASSESSMENT GUIDELINES:
1. **CLINICAL REASONING (90% AI Intelligence)**: Use your therapeutic expertise to deeply understand the context
2. **DEVELOPMENTAL APPROPRIATENESS**: Consider what a {age}-year-old with {diagnosis} typically experiences
3. **CONTEXTUAL ANALYSIS**: Reference their routine (e.g., if near snack time and sad, consider hunger/blood sugar)
4. **RESOURCE AWARENESS (10% Database)**: Consider their favorite activities, comfort items, and preferences
5. **CONDITION-SPECIFIC THINKING**: Account for common challenges and triggers for children with {diagnosis}
6. **CHILD-CENTERED LANGUAGE**: Make prompts simple, clear, and age-appropriate for a {age}-year-old
7. **PERSONALIZATION**: Use the child's name ({child_name}) to build rapport and trust
8. **ROOT CAUSE IDENTIFICATION**: Think like a therapist - what underlying needs or problems could be causing this emotion?
9. **SOLUTION-FOCUSED APPROACH**: Choose prompts that, when explored, will lead to clear, actionable interventions
10. **CLINICAL PRIORITIZATION**: Start with the most likely and easily addressable issues based on evidence and experience

FORMAT:
Return ONLY a JSON array with exactly 4 objects:
[
  {{
    "label": "Short label (2-4 words)",
    "description": "Brief, child-friendly description (one sentence)",
    "reasoning": "Why this is relevant given the context"
  }}
]

CLINICAL QUALITY CHECK:
Before finalizing, ask yourself as a therapist:
- Will exploring these prompts lead to a clear, solvable problem?
- Are these the most likely causes based on clinical experience and the context?
- Can a caregiver implement immediate, evidence-based interventions once the root cause is identified?
- Are these developmentally appropriate for the child's age and diagnosis?
- Do these align with best practices in pediatric behavioral therapy?

Generate the 4 most clinically relevant and solution-oriented prompts now:"""

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
        
        prompt = f"""You are a pediatric behavioral therapist helping {child_name} ({age} years old, {diagnosis}) communicate why they're feeling {emotion}.

USE YOUR THERAPEUTIC EXPERTISE to dig deeper systematically, like you would in a clinical assessment.

CONVERSATION SO FAR:
{history_text}

CONTEXT:
- Current Time: {current_time}
- Diagnosis: {diagnosis}
- Favorite Activities: {routine.get('favorite_activities', 'Not specified')}
- Comfort Items: {routine.get('comfort_items', 'Not specified')}

{child_name} just selected "{selected_option}" as a reason for feeling {emotion}.

TASK:
Use your therapeutic expertise to generate 4 specific follow-up prompts that dig deeper into WHY they selected "{selected_option}".

THERAPEUTIC ASSESSMENT GUIDELINES:
1. **CLINICAL SPECIFICITY**: Be very specific and actionable, like you would in a therapy session
2. **DIAGNOSIS-INFORMED**: Consider how {diagnosis} affects the child's experience and expression
3. **DEVELOPMENTAL LENS**: Think about what a {age}-year-old with {diagnosis} typically experiences
4. **CONTEXTUAL INTEGRATION**: Reference their routine and preferences when clinically relevant
5. **CHILD-FRIENDLY CLARITY**: Make it easy for the child to identify the exact problem
6. **DIFFERENTIAL DIAGNOSIS**: Each option should help narrow down to the specific, solvable issue
7. **SYSTEMATIC ELIMINATION**: Ask questions that eliminate possibilities and pinpoint the exact problem (like a clinical assessment)
8. **INTERVENTION-READY**: Choose prompts that, when selected, will make evidence-based interventions obvious and actionable

FORMAT:
Return ONLY a JSON array with exactly 4 objects:
[
  {{
    "label": "Specific reason (3-5 words)",
    "description": "Clear, simple description",
    "reasoning": "Why this makes sense given {selected_option}"
  }}
]

CLINICAL QUALITY CHECK:
Before finalizing, ask yourself as a therapist:
- Will selecting one of these lead to a specific, evidence-based intervention?
- Do these narrow down the problem effectively using clinical reasoning?
- Are these the most likely specific causes of "{selected_option}" based on therapeutic experience?
- Can each option be clearly resolved with concrete, actionable interventions?
- Do these follow best practices in pediatric behavioral assessment?

Generate the 4 most clinically diagnostic and intervention-enabling follow-up prompts:"""

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
        
        prompt = f"""You are a pediatric behavioral therapist providing an evidence-based intervention plan for {child_name} ({age} years old, {diagnosis}).

CLINICAL ASSESSMENT COMPLETE:
PROBLEM IDENTIFIED:
{child_name} is feeling {emotion} because: {history_text}

CONTEXT:
- Diagnosis: {diagnosis}
- Age: {age} years old
- Favorite Activities: {routine.get('favorite_activities', 'Not specified')}
- Comfort Items: {routine.get('comfort_items', 'Not specified')}

TASK:
Generate the BEST POSSIBLE evidence-based intervention that will effectively resolve {child_name}'s problem.

THERAPEUTIC INTERVENTION REQUIREMENTS:
1. **EMOTIONAL VALIDATION**: Start by validating {child_name}'s emotion (therapeutic rapport-building)
2. **CLINICAL FORMULATION**: Clearly state what the actual problem is based on your assessment
3. **EVIDENCE-BASED INTERVENTIONS**: Provide 2-4 concrete, actionable steps based on best practices in pediatric therapy
4. **CLINICAL EFFECTIVENESS**: Focus on interventions proven to work, not just what sounds nice
5. **DEVELOPMENTAL APPROPRIATENESS**: Use language and interventions suitable for a {age}-year-old
6. **DIAGNOSIS-INFORMED**: Account for how {diagnosis} affects the child's needs, abilities, and response to interventions
7. **RESOURCE UTILIZATION**: Reference their comfort items, favorite activities, or routine as therapeutic tools when clinically appropriate
8. **IMMEDIATE IMPLEMENTATION**: Provide interventions that can be implemented right now by caregivers
9. **CLEAR ACTION STEPS**: Make it obvious what needs to happen next (like a treatment plan)
10. **POSITIVE REINFORCEMENT**: End with therapeutic encouragement and hope

INTERVENTION PLAN STRUCTURE:
- First sentence: Validate emotion and state clinical formulation
- Middle: 2-4 specific, evidence-based intervention steps (numbered or bulleted)
- Last sentence: Therapeutic encouragement and positive prognosis

Keep it under 120 words. Use warm, simple, direct language (therapeutic but not clinical-sounding).

Generate the most effective, evidence-based intervention now:"""

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
