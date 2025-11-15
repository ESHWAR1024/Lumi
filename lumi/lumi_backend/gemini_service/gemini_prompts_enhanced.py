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

    # ---------------- FIXED FIRST-LEVEL CARDS ---------------- #

    def _get_fixed_initial_prompts(
        self,
        emotion: str,
        child_name: str,
        age: int
    ) -> List[Dict[str, str]]:
        """
        For ALL emotions, the very first 4 cards are FIXED.
        For core emotions (fear, happy, sad, angry, surprise), we use emotion-specific cards.
        For other emotions, we use a generic set.
        After one of these is selected, AI takes over and digs deeper.
        """
        e = (emotion or "").strip().lower()

        if e == "fear" or e == "scared":
            return [
                {
                    "label": "Scared of something",
                    "description": f"{child_name}, are you scared of something around you?",
                    "reasoning": "Common reason for feeling fear is a specific thing nearby that feels scary."
                },
                {
                    "label": "Too much noise",
                    "description": f"{child_name}, is it too loud or noisy?",
                    "reasoning": "Loud sounds can make children feel afraid or overwhelmed, especially with sensory differences."
                },
                {
                    "label": "Dark or alone",
                    "description": f"{child_name}, are you scared of the dark or being alone?",
                    "reasoning": "Darkness or being away from safe people often triggers fear in children."
                },
                {
                    "label": "Bad dream or thought",
                    "description": f"{child_name}, did a bad dream or thought make you scared?",
                    "reasoning": "Unpleasant dreams or thoughts can keep fear going even when nothing is happening right now."
                },
            ]

        if e == "happy":
            return [
                {
                    "label": "Fun activity",
                    "description": f"{child_name}, are you happy because you are doing something fun?",
                    "reasoning": "Enjoyable activities are a very common cause of happiness in children."
                },
                {
                    "label": "With favorite person",
                    "description": f"{child_name}, are you happy because you are with someone you like?",
                    "reasoning": "Being with trusted people brings joy and emotional safety."
                },
                {
                    "label": "Got something nice",
                    "description": f"{child_name}, are you happy because you got something you like?",
                    "reasoning": "Receiving toys, treats, or praise can quickly boost happiness."
                },
                {
                    "label": "Proud of yourself",
                    "description": f"{child_name}, are you happy because you did something all by yourself?",
                    "reasoning": "Children often feel happy when they achieve something or feel capable."
                },
            ]

        if e == "sad":
            return [
                {
                    "label": "Missing someone",
                    "description": f"{child_name}, are you sad because you miss someone?",
                    "reasoning": "Separation from important people is a common cause of sadness."
                },
                {
                    "label": "Something went wrong",
                    "description": f"{child_name}, are you sad because something did not go how you wanted?",
                    "reasoning": "Disappointment or a small failure often leads to sadness in children."
                },
                {
                    "label": "Feeling left out",
                    "description": f"{child_name}, are you sad because you feel left out?",
                    "reasoning": "Social exclusion or not being included in play can cause sadness."
                },
                {
                    "label": "Body not okay",
                    "description": f"{child_name}, are you sad because your body does not feel good?",
                    "reasoning": "Pain, tiredness, or discomfort can also show up as sadness in children."
                },
            ]

        if e == "angry" or e == "mad":
            return [
                {
                    "label": "Someone upset you",
                    "description": f"{child_name}, are you angry because someone said or did something?",
                    "reasoning": "Conflict or feeling treated unfairly commonly triggers anger."
                },
                {
                    "label": "Things not working",
                    "description": f"{child_name}, are you angry because something is not working how you want?",
                    "reasoning": "Frustration with tasks or toys not working can cause anger, especially with communication challenges."
                },
                {
                    "label": "Too many rules",
                    "description": f"{child_name}, are you angry because you were told ‘no’ or had to stop?",
                    "reasoning": "Limits, rules, or ending a favorite activity are common anger triggers."
                },
                {
                    "label": "Body feels tight",
                    "description": f"{child_name}, are you angry because your body feels tight or uncomfortable?",
                    "reasoning": "Physical discomfort or sensory overload can feel like anger inside the body."
                },
            ]

        if e == "surprise" or e == "surprised":
            return [
                {
                    "label": "Good surprise",
                    "description": f"{child_name}, are you surprised in a happy way?",
                    "reasoning": "Some surprises feel exciting and positive."
                },
                {
                    "label": "Scary surprise",
                    "description": f"{child_name}, are you surprised in a scared way?",
                    "reasoning": "Unexpected events can feel shocking or frightening."
                },
                {
                    "label": "Plans changed suddenly",
                    "description": f"{child_name}, are you surprised because plans changed?",
                    "reasoning": "Children, especially with neurodivergence, can feel surprised or unsettled by sudden routine changes."
                },
                {
                    "label": "New thing you saw",
                    "description": f"{child_name}, are you surprised because you saw something new?",
                    "reasoning": "New people, objects, or places can cause surprise and curiosity."
                },
            ]

        # Generic fixed starting cards for any other emotion
        return [
            {
                "label": "Body feels different",
                "description": f"{child_name}, do you feel something strange or different in your body?",
                "reasoning": "Many emotions come with body changes like tightness, heaviness, or butterflies."
            },
            {
                "label": "Need something now",
                "description": f"{child_name}, do you think you need something right now?",
                "reasoning": "Basic needs (food, drink, rest, bathroom) often sit underneath strong emotions."
            },
            {
                "label": "Something happened today",
                "description": f"{child_name}, did something happen today that made you feel this way?",
                "reasoning": "Events at home, school, or therapy often trigger emotional changes."
            },
            {
                "label": "Someone made you feel this",
                "description": f"{child_name}, did another person make you feel this way?",
                "reasoning": "Interactions with others are a frequent source of emotional reactions."
            },
        ]
    
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
        FIRST LEVEL:
        For ALL emotions, we show 4 FIXED cards at the start.
        For fear/happy/sad/angry/surprise, these are emotion-specific.
        For other emotions, we show 4 generic fixed cards.

        AFTER the child selects one of these cards, THEN the AI
        (generate_followup_prompts) comes into play to dig deeper.
        """
        # Always use fixed first-level cards now
        return self._get_fixed_initial_prompts(emotion, child_name, age)
    
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
        The new prompts must be strictly more specific "children" of the selected_option.
        This is where the AI comes into play AFTER the fixed first-level cards.
        """
        
        # Build conversation path for context
        if conversation_history:
            history_path = " → ".join(conversation_history) + f" → {selected_option}"
            previous_context = f"\nPrevious selections: {' → '.join(conversation_history)}"
        else:
            history_path = selected_option
            previous_context = "\nPrevious selections: None (this is the first selection)"
        
        prompt = f"""You are an expert pediatric behavioral therapist conducting a clinical assessment for {child_name} ({age} years old, {diagnosis}).

YOUR MISSION: Generate 4 SPECIFIC, DEEPER follow-up prompts to dig into WHY {child_name} selected "{selected_option}".
You must treat "{selected_option}" as a BROAD CATEGORY and each new card as a MORE DETAILED POSSIBILITY inside that category.

Think of the conversation as a decision tree:
- First cards = big branches (FIXED by the system for all emotions)
- Selected card = chosen branch
- YOUR 4 new cards = smaller branches under that SAME branch (no new branches)

═══════════════════════════════════════════════════════════════════
📊 CONVERSATION PATH (for context only):
═══════════════════════════════════════════════════════════════════
Full path: {history_path}
{previous_context}

⚠️ IMPORTANT: Generate follow-ups for ONLY the most recent selection: "{selected_option}"
(The previous selections provide context, but your 4 prompts MUST dig deeper into "{selected_option}" specifically.)

You are NOT allowed to:
- Introduce brand-new, unrelated reasons that are not clearly part of "{selected_option}"
- Repeat any labels or descriptions already used earlier in the conversation
- Jump back to other top-level causes (e.g., hunger, tired, lonely) if they are different from "{selected_option}"

CURRENT FOCUS: "{selected_option}"
EMOTION: {emotion}

═══════════════════════════════════════════════════════════════════
📊 CHILD PROFILE & ROUTINE DATA:
═══════════════════════════════════════════════════════════════════
- Current Time: {current_time}
- Age: {age} years old
- Diagnosis: {diagnosis}

ROUTINE TIMES:
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

═══════════════════════════════════════════════════════════════════
🧠 USE YOUR 90% BIG BRAIN TO ANALYZE "{selected_option}" DEEPLY:
═══════════════════════════════════════════════════════════════════

CLINICAL DIFFERENTIAL DIAGNOSIS:
What are the TOP 4 MOST SPECIFIC, REAL-WORLD reasons a {age}-year-old with {diagnosis} would pick "{selected_option}"?

Each new option must answer the hidden question:
"Okay, you feel '{selected_option}'. Is it because of THIS, or THIS, or THIS, or THIS?"

IF "{selected_option}" relates to HUNGER/FOOD:
→ Narrow down to specific versions of hunger:
   - Missed snack or meal at this time?
   - Wanting a specific favorite food?
   - Sensory/texture issues making current food hard to eat?
   - Thirst vs hunger?
→ All 4 options must be concrete, hunger-related variations.

IF "{selected_option}" relates to TIREDNESS/FATIGUE:
→ Narrow down to specific types of tired:
   - Sleepy because it's near bedtime?
   - Physically tired from too much movement (especially with {diagnosis})?
   - Tired from school/therapy or thinking too hard?
   - Body pain or muscle fatigue?
→ All 4 options must stay inside "tired" but get more detailed.

IF "{selected_option}" relates to ACTIVITIES/PLAY:
→ Narrow down to specific activity needs:
   - Wanting one of the favorite_activities listed in the database?
   - Wanting someone specific to play with?
   - Bored with current activity and want a different one?
   - Wanting quiet play vs active play?
→ All 4 options must stay inside "want to do something", but be more precise.

IF "{selected_option}" relates to DISCOMFORT/PAIN:
→ Narrow down to specific discomfort:
   - Positioning issues (especially for {diagnosis})?
   - Clothing, noise, light, temperature, or touch sensitivity?
   - Needing bathroom or diaper change?
   - Headache, tummy ache, or body pain?
→ All 4 options must be distinct, concrete discomfort explanations.

IF "{selected_option}" relates to EMOTIONAL NEEDS:
→ Narrow down to specific emotional needs:
   - Wanting a comfort_item (teddy, blanket, etc.)?
   - Wanting a hug or attention from a specific person?
   - Feeling scared of something specific?
   - Upset because of a change in routine?
→ All 4 options must be emotional sub-reasons, not new top-level causes.

═══════════════════════════════════════════════════════════════════
🔎 DIVE-DEEPER LOGIC (MANDATORY):
═══════════════════════════════════════════════════════════════════

For EACH of the 4 follow-up prompts:
- It MUST be a "child" reason under "{selected_option}"
- It MUST add a NEW layer of detail (more specific than before)
- It MUST be observable in real life (something a caregiver can check or act on)
- It MUST be short, concrete, and easy for a child to understand

Examples of "deepening" (not new topics):
- Selected: "Hungry" → Follow-ups like:
  - "Missed your snack?"
  - "Want your favorite food?"
- Selected: "Tired" → Follow-ups like:
  - "Sleepy because it's late?"
  - "Body hurts from moving?"

Do NOT do this:
- Selected: "Hungry" → Follow-ups like "Feeling lonely?" or "Too noisy here?"
(Those are new branches, not deeper parts of hunger.)

═══════════════════════════════════════════════════════════════════
✅ MANDATORY REQUIREMENTS FOR FOLLOW-UP PROMPTS:
═══════════════════════════════════════════════════════════════════

1. **BE HYPER-SPECIFIC**:
   - Don't ask "Why tired?" → Ask "Tired from playing too much?" or "Need to sleep now?"
2. **USE TIME CONTEXT**:
   - Reference current_time vs routine times when helpful (e.g., "because it's near snack time?")
3. **USE PREFERENCES**:
   - Reference their actual favorite_activities or comfort_items when they logically fit under "{selected_option}"
4. **USE DIAGNOSIS KNOWLEDGE**:
   - Include {diagnosis}-specific factors (fatigue, sensory overload, communication difficulty, etc.)
5. **LEAD TO SOLUTIONS**:
   - Each prompt should point toward a concrete, fixable thing a caregiver can do
6. **STAY INSIDE THE BRANCH**:
   - Every card must clearly feel like "a type of {selected_option}", not something else

═══════════════════════════════════════════════════════════════════
💡 SMART FOLLOW-UP EXAMPLES:
═══════════════════════════════════════════════════════════════════

Example 1: Selected "Tired"
- Current time: 7:30 PM, Bedtime: 8:00 PM
- Diagnosis: Cerebral Palsy
→ Follow-ups (all are "kinds of tired"):
  1. "Time for bed?" (Near bedtime)
  2. "Body feeling sore?" (CP causes muscle fatigue)
  3. "Too tired from moving?" (Overexertion)
  4. "Eyes feeling heavy?" (Sleepiness)

Example 2: Selected "Hungry"
- Current time: 3:00 PM, Snack time: 3:00 PM
- Favorite activities: "eating cookies"
→ Follow-ups (all are "kinds of hungry"):
  1. "Want your snack now?" (Time match)
  2. "Want cookies?" (Favorite food)
  3. "Thirsty? Need water?" (Related basic need)
  4. "Tummy hurting from hunger?" (Hunger discomfort)

Example 3: Selected "Want to play"
- Favorite activities: "listening to stories, playing with toys"
→ Follow-ups (all are "ways to play"):
  1. "Want to hear a story?" (Specific favorite activity)
  2. "Want to play with toys?" (Specific favorite activity)
  3. "Want someone to play with?" (Social need inside play)
  4. "Bored with this game?" (Reason for wanting play change)

═══════════════════════════════════════════════════════════════════
📋 OUTPUT FORMAT:
═══════════════════════════════════════════════════════════════════

Return ONLY a JSON array with exactly 4 objects:
[
  {{
    "label": "Specific reason (3-5 words)",
    "description": "Clear, simple description",
    "reasoning": "Why this makes sense given '{selected_option}' and stays inside that category"
  }}
]

═══════════════════════════════════════════════════════════════════
🎯 GENERATE NOW:
═══════════════════════════════════════════════════════════════════

Generate 4 HYPER-SPECIFIC follow-up prompts that:
- Are ALL children of "{selected_option}"
- Go one level deeper into the real cause
- Help identify the EXACT problem so we can provide the perfect solution.

Generate the 4 follow-up prompts:"""

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
        
        # Build the complete problem identification from conversation
        if conversation_history:
            problem_path = " → ".join(conversation_history)
            final_problem = conversation_history[-1]  # The most specific problem identified
        else:
            problem_path = "Unknown reason"
            final_problem = "Unknown reason"
        
        prompt = f"""You are an expert pediatric behavioral therapist creating a PERSONALIZED intervention plan for {child_name} ({age} years old, {diagnosis}).

═══════════════════════════════════════════════════════════════════
✅ CLINICAL ASSESSMENT COMPLETE - PROBLEM IDENTIFIED:
═══════════════════════════════════════════════════════════════════

EMOTION: {emotion}

PROBLEM DISCOVERY PATH: {problem_path}

ROOT CAUSE IDENTIFIED: {final_problem}

Summary: {child_name} is feeling {emotion} because {final_problem}

═══════════════════════════════════════════════════════════════════
📊 CHILD PROFILE & ROUTINE DATA FOR PERSONALIZED SOLUTION:
═══════════════════════════════════════════════════════════════════

CHILD PROFILE:
- Name: {child_name}
- Age: {age} years old
- Diagnosis: {diagnosis}
- Current Time: {routine.get('current_time', 'Not specified')}

ROUTINE TIMES:
- Wake up: {routine.get('wake_up_time', 'Not specified')}
- Breakfast: {routine.get('breakfast_time', 'Not specified')}
- Lunch: {routine.get('lunch_time', 'Not specified')}
- Snacks: {routine.get('snacks_time', 'Not specified')}
- Dinner: {routine.get('dinner_time', 'Not specified')}
- Bedtime: {routine.get('bedtime', 'Not specified')}

PERSONAL PREFERENCES:
- Favorite Activities: {routine.get('favorite_activities', 'Not specified')}
- Comfort Items: {routine.get('comfort_items', 'Not specified')}
- Communication Style: {routine.get('communication_preferences', 'Not specified')}

═══════════════════════════════════════════════════════════════════
🧠 USE YOUR 90% BIG BRAIN TO CREATE PERFECT SOLUTION:
═══════════════════════════════════════════════════════════════════

STEP 1: ANALYZE THE ROOT CAUSE
→ What is the EXACT problem based on the conversation history?
→ How does {diagnosis} affect this specific situation?
→ What does a {age}-year-old with {diagnosis} need right now?

STEP 2: CREATE TIME-AWARE SOLUTION
→ If problem is hunger + near meal time → Provide specific food/meal
→ If problem is fatigue + near bedtime → Initiate bedtime routine
→ If problem is boredom + has favorite activities → Suggest specific activity

STEP 3: INTEGRATE THEIR PREFERENCES
→ Use their actual favorite_activities in the solution
→ Use their actual comfort_items for emotional regulation
→ Match their communication_preferences style

STEP 4: MAKE IT DIAGNOSIS-SPECIFIC
→ For Cerebral Palsy: Consider positioning, physical comfort, mobility assistance
→ For Autism: Consider sensory needs, routine, predictability
→ For Down Syndrome: Consider communication support, patience, encouragement
→ Use evidence-based interventions for {diagnosis}

═══════════════════════════════════════════════════════════════════
✅ MANDATORY PERSONALIZATION REQUIREMENTS:
═══════════════════════════════════════════════════════════════════

YOUR SOLUTION MUST:

1. **USE {child_name}'S NAME** multiple times (build rapport)
2. **REFERENCE SPECIFIC DATABASE DATA**:
   - If favorite_activities exists → Include it in solution
   - If comfort_items exists → Suggest using it
   - If near routine time → Reference that specific routine
3. **BE DIAGNOSIS-SPECIFIC**: Include interventions proven for {diagnosis}
4. **BE IMMEDIATELY ACTIONABLE**: Caregiver can do this RIGHT NOW
5. **BE CONCRETE**: No vague advice - specific steps with specific items/activities

═══════════════════════════════════════════════════════════════════
💡 PERSONALIZED SOLUTION EXAMPLES:
═══════════════════════════════════════════════════════════════════

Example 1: Hungry + Snack Time + Favorite Activity "eating cookies"
→ "I understand you're hungry, Mia! It's snack time. Let's get you some cookies (your favorite!) and a drink. Sit in your comfy spot and enjoy your snack. You'll feel much better!"

Example 2: Tired + Cerebral Palsy + Comfort Item "teddy bear"
→ "I understand you're tired, Alex. Your body works hard! Let's get your teddy bear, adjust your position so you're comfy, and rest. It's almost bedtime anyway. You deserve this rest!"

Example 3: Bored + Favorite Activity "listening to stories"
→ "I understand you're bored, Emma! Let's do something fun. How about we listen to one of your favorite stories? Get comfy with your blanket and let's start story time!"

═══════════════════════════════════════════════════════════════════
📋 SOLUTION STRUCTURE (Keep under 120 words):
═══════════════════════════════════════════════════════════════════

1. **VALIDATION** (1 sentence):
   "I understand you're {emotion}, {child_name}, because [specific problem from history]"

2. **PERSONALIZED SOLUTION** (2-4 action steps):
   - Step 1: Immediate action (use specific favorite_activities or comfort_items)
   - Step 2: Supporting action (reference routine or diagnosis needs)
   - Step 3: Additional comfort (if needed)

3. **ENCOURAGEMENT** (1 sentence):
   Positive, hopeful message using {child_name}'s name

═══════════════════════════════════════════════════════════════════
🎯 GENERATE NOW:
═══════════════════════════════════════════════════════════════════

Create the MOST EFFECTIVE, PERSONALIZED solution for {child_name} using:
- Their specific problem from conversation
- Their favorite_activities and comfort_items
- Their routine and current time
- Evidence-based interventions for {diagnosis}
- Age-appropriate language for {age}-year-old

Generate the personalized solution (under 120 words):"""

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
        """Fallback prompts if API fails (not used for first-level anymore, but kept as backup)."""
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
