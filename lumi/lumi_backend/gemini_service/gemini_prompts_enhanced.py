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

DAILY ROUTINE ANALYSIS:
- Wake up: {routine.get('wake_up_time', 'Not specified')}
- Breakfast: {routine.get('breakfast_time', 'Not specified')}
- Lunch: {routine.get('lunch_time', 'Not specified')}
- Snacks: {routine.get('snacks_time', 'Not specified')}
- Dinner: {routine.get('dinner_time', 'Not specified')}
- Bedtime: {routine.get('bedtime', 'Not specified')}

PERSONAL PREFERENCES & TRIGGERS:
- Favorite Activities: {routine.get('favorite_activities', 'Not specified')}
- Comfort Items: {routine.get('comfort_items', 'Not specified')}
- Preferred Communication: {routine.get('preferred_prompts', 'Not specified')}
- Communication Preferences: {routine.get('communication_preferences', 'Not specified')}

CURRENT SITUATION:
{child_name} is feeling {emotion} right now at {current_time}.

SMART ANALYSIS REQUIRED:
Use 90% of your clinical intelligence to analyze:
- Is it near meal/snack time? (Check current time vs routine)
- What activities does {child_name} love? (Reference favorite_activities)
- What comforts them? (Reference comfort_items)
- How does {diagnosis} typically affect children at age {age}?
- What are common triggers for {emotion} in children with {diagnosis}?
- What time-specific needs might {child_name} have right now?

TASK:
Generate exactly 4 picture prompts that could explain why {child_name} is feeling {emotion}.

ENHANCED THERAPEUTIC GUIDELINES:
1. **CLINICAL INTELLIGENCE (90%)**: Use deep therapeutic knowledge about {diagnosis} in {age}-year-olds
2. **TIME-AWARE ANALYSIS**: If current time matches snack_time/meal_time, strongly consider hunger
3. **ACTIVITY-BASED REASONING**: If {child_name} loves specific activities, consider if they want/miss them
4. **COMFORT-SEEKING BEHAVIOR**: Reference their comfort_items when they're distressed
5. **DIAGNOSIS-SPECIFIC PATTERNS**: Children with {diagnosis} have specific needs - address them
6. **ROUTINE DISRUPTION ANALYSIS**: Consider if normal routine might be disrupted
7. **DEVELOPMENTAL STAGE**: What does a {age}-year-old typically need when feeling {emotion}?
8. **PERSONALIZED LANGUAGE**: Use {child_name}'s name and reference their specific preferences
9. **ROOT CAUSE FOCUS**: Don't just address symptoms - find underlying needs
10. **EVIDENCE-BASED PRIORITIZATION**: Most likely causes first, based on time, routine, and condition

EXAMPLE SMART REASONING:
If it's 6pm and {child_name}'s snack_time is 6pm and they're sad → HIGH PROBABILITY: Hungry
If {child_name} loves "listening to stories" and they're bored → HIGH PROBABILITY: Want story time
If {child_name} has cerebral palsy and it's evening → CONSIDER: Physical fatigue, positioning needs
If {child_name}'s comfort_item is "teddy bear" and they're anxious → CONSIDER: Want comfort item

FORMAT:
Return ONLY a JSON array with exactly 4 objects:
[
  {{
    "label": "Short label (2-4 words)",
    "description": "Brief, child-friendly description (one sentence)",
    "reasoning": "Why this is relevant given the context"
  }}
]

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

COMPREHENSIVE CHILD CONTEXT:
- Current Time: {current_time}
- Age: {age} years old
- Diagnosis: {diagnosis}
- Favorite Activities: {routine.get('favorite_activities', 'Not specified')}
- Comfort Items: {routine.get('comfort_items', 'Not specified')}
- Preferred Communication: {routine.get('preferred_prompts', 'Not specified')}

ROUTINE ANALYSIS:
- Wake up: {routine.get('wake_up_time', 'Not specified')}
- Breakfast: {routine.get('breakfast_time', 'Not specified')}
- Lunch: {routine.get('lunch_time', 'Not specified')}
- Snacks: {routine.get('snacks_time', 'Not specified')}
- Dinner: {routine.get('dinner_time', 'Not specified')}
- Bedtime: {routine.get('bedtime', 'Not specified')}

{child_name} just selected "{selected_option}" as a reason for feeling {emotion}.

SMART FOLLOW-UP ANALYSIS:
Use 90% clinical intelligence to analyze WHY {child_name} chose "{selected_option}":

If "{selected_option}" = "Hungry":
- Is it near their meal/snack time? (Check current_time vs routine)
- Do they have specific food preferences/aversions due to {diagnosis}?
- Could it be blood sugar related? Medication timing?

If "{selected_option}" = "Tired":
- Is it near bedtime? Physical exhaustion from {diagnosis}?
- Medication side effects? Overstimulation? Need position change?

If "{selected_option}" = "Want to play":
- What specific activities do they love? (Reference favorite_activities)
- Are they bored? Missing social interaction? Need sensory input?

TASK:
Generate 4 specific follow-up prompts that dig deeper into WHY they selected "{selected_option}".

ENHANCED THERAPEUTIC GUIDELINES:
1. **CLINICAL INTELLIGENCE (90%)**: Use deep knowledge about {diagnosis} and {age}-year-old development
2. **SPECIFIC ROOT CAUSES**: Don't ask "why tired?" - ask "hungry because missed snack?" or "tired from medication?"
3. **TIME-CONTEXTUAL**: Reference current time vs their routine for relevant prompts
4. **CONDITION-SPECIFIC**: Address common issues for children with {diagnosis}
5. **ACTIONABLE SPECIFICITY**: Each prompt should lead to a clear, specific solution
6. **ROUTINE-INTEGRATED**: Reference their actual meal times, activity preferences, comfort items
7. **DEVELOPMENTAL APPROPRIATE**: What would a {age}-year-old with {diagnosis} specifically experience?
8. **INTERVENTION-FOCUSED**: Each option should point to a concrete action/solution

FORMAT:
Return ONLY a JSON array with exactly 4 objects:
[
  {{
    "label": "Specific reason (3-5 words)",
    "description": "Clear, simple description",
    "reasoning": "Why this makes sense given {selected_option}"
  }}
]

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

COMPREHENSIVE CONTEXT FOR SMART SOLUTION:
- Child: {child_name}, Age: {age}, Diagnosis: {diagnosis}
- Current Time: {routine.get('current_time', 'Not specified')}
- Favorite Activities: {routine.get('favorite_activities', 'Not specified')}
- Comfort Items: {routine.get('comfort_items', 'Not specified')}
- Communication Preferences: {routine.get('communication_preferences', 'Not specified')}

ROUTINE CONTEXT:
- Wake up: {routine.get('wake_up_time', 'Not specified')}
- Breakfast: {routine.get('breakfast_time', 'Not specified')}
- Lunch: {routine.get('lunch_time', 'Not specified')}
- Snacks: {routine.get('snacks_time', 'Not specified')}
- Dinner: {routine.get('dinner_time', 'Not specified')}
- Bedtime: {routine.get('bedtime', 'Not specified')}

TASK:
Generate the MOST EFFECTIVE, PERSONALIZED intervention that will resolve {child_name}'s specific problem.

USE 90% CLINICAL INTELLIGENCE:
- What does {child_name} specifically need based on their {diagnosis}?
- How does their age ({age}) affect the best intervention approach?
- What time-specific solutions are needed? (Reference current time vs routine)
- How can their favorite_activities and comfort_items be used therapeutically?
- What are evidence-based interventions for {diagnosis} in {age}-year-olds?

ENHANCED INTERVENTION REQUIREMENTS:
1. **PERSONALIZED VALIDATION**: "I understand you're {emotion}, {child_name}, because [specific problem]"
2. **SMART PROBLEM IDENTIFICATION**: Reference the actual root cause from conversation
3. **CONDITION-SPECIFIC INTERVENTIONS**: Use proven strategies for {diagnosis}
4. **TIME-AWARE SOLUTIONS**: If it's meal time → food solutions, if bedtime → sleep solutions
5. **RESOURCE-INTEGRATED**: Use their actual comfort_items and favorite_activities
6. **DEVELOPMENTALLY PRECISE**: Perfect for a {age}-year-old with {diagnosis}
7. **IMMEDIATELY ACTIONABLE**: Caregivers can do this right now
8. **ROUTINE-ALIGNED**: Work with their established schedule and preferences
9. **EVIDENCE-BASED**: Proven therapeutic techniques
10. **ENCOURAGING**: Build confidence and hope

SMART SOLUTION EXAMPLES:
- If hungry + snack_time → "Get your favorite snack (reference their preferences) and sit in your comfort spot"
- If tired + cerebral palsy → "Let's adjust your position, use your comfort_item, and rest in your favorite way"
- If want story + favorite_activity is stories → "Let's read your favorite story with your comfort_item"

INTERVENTION STRUCTURE:
- Validation: "I understand you're {emotion}, {child_name}, because [specific problem]"
- Solution: 2-4 specific, personalized action steps using their context
- Encouragement: Positive, hopeful ending

Keep it under 120 words. Use {child_name}'s name and reference their specific context.

Generate the most effective, personalized intervention now:"""

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
