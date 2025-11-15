"""
FastAPI service for emotion-based communication prompts.
Enhanced with context-aware AI using child profile and routine data.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
from dotenv import load_dotenv
from gemini_prompts_enhanced import EnhancedGeminiService
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Lumi Communication Service", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Enhanced Gemini service with multiple API keys
GEMINI_API_KEYS = [
    key.strip() 
    for key in os.environ.get('GEMINI_API_KEYS', '').split(',') 
    if key.strip()
]

if not GEMINI_API_KEYS:
    print("⚠️ Warning: No Gemini API keys found. Set GEMINI_API_KEYS environment variable.")
    gemini_service = None
else:
    gemini_service = EnhancedGeminiService(GEMINI_API_KEYS)
    print(f"✅ Enhanced Gemini service initialized with {len(GEMINI_API_KEYS)} API key(s)")


# Request/Response models
class ChildProfile(BaseModel):
    child_name: str
    age: int
    diagnosis: str | None = None


class ChildRoutine(BaseModel):
    wake_up_time: str | None = None
    breakfast_time: str | None = None
    lunch_time: str | None = None
    snacks_time: str | None = None
    dinner_time: str | None = None
    bedtime: str | None = None
    favorite_activities: str | None = None
    comfort_items: str | None = None
    preferred_prompts: str | None = None
    communication_preferences: str | None = None


class InitialPromptsRequest(BaseModel):
    emotion: str
    child_profile_id: str
    confidence: float
    child_profile: ChildProfile
    child_routine: ChildRoutine
    current_time: str  # Format: "HH:MM"


class FollowupPromptsRequest(BaseModel):
    session_id: str
    emotion: str
    selected_option: str
    child_profile_id: str
    child_profile: ChildProfile
    child_routine: ChildRoutine
    current_time: str
    interaction_depth: int
    conversation_history: list[str]  # Previous selections


class DigDeeperRequest(BaseModel):
    session_id: str
    emotion: str
    conversation_history: list[str]
    child_profile: ChildProfile
    child_routine: ChildRoutine
    current_time: str


class SolutionRequest(BaseModel):
    session_id: str
    emotion: str
    conversation_history: list[str]
    child_profile: ChildProfile
    child_routine: ChildRoutine


class RegenerateSolutionRequest(BaseModel):
    session_id: str
    emotion: str
    conversation_history: list[str]
    previous_solution: str
    child_profile: ChildProfile
    child_routine: ChildRoutine


class PromptOption(BaseModel):
    label: str
    description: str
    reasoning: str | None = None


class PromptsResponse(BaseModel):
    session_id: Optional[str]
    prompts: List[PromptOption]
    prompt_type: str  # 'initial', 'followup', 'dig_deeper'


class SolutionResponse(BaseModel):
    session_id: str
    solution: str
    emotion: str


@app.get('/')
def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Lumi Communication Service",
        "version": "2.0.0",
        "gemini_configured": gemini_service is not None
    }


@app.post('/api/prompts/initial', response_model=PromptsResponse)
async def get_initial_prompts(request: InitialPromptsRequest):
    """
    Get initial 4-5 context-aware picture prompts based on detected emotion.
    Uses child profile, routine, and current time for intelligent suggestions.
    """
    if gemini_service is None:
        raise HTTPException(status_code=503, detail="Gemini service not configured")
    
    try:
        # Generate context-aware prompts using Enhanced Gemini
        prompts = gemini_service.generate_initial_prompts(
            emotion=request.emotion,
            child_name=request.child_profile.child_name,
            age=request.child_profile.age,
            diagnosis=request.child_profile.diagnosis or "Not specified",
            routine={
                'wake_up_time': request.child_routine.wake_up_time,
                'breakfast_time': request.child_routine.breakfast_time,
                'lunch_time': request.child_routine.lunch_time,
                'snacks_time': request.child_routine.snacks_time,
                'dinner_time': request.child_routine.dinner_time,
                'bedtime': request.child_routine.bedtime,
                'favorite_activities': request.child_routine.favorite_activities,
                'comfort_items': request.child_routine.comfort_items,
                'preferred_prompts': request.child_routine.preferred_prompts
            },
            current_time=request.current_time
        )
        
        # Generate session ID
        import uuid
        session_id = str(uuid.uuid4())
        
        return PromptsResponse(
            session_id=session_id,
            prompts=[PromptOption(**p) for p in prompts],
            prompt_type="initial"
        )
    
    except Exception as e:
        print(f"❌ Error generating initial prompts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate prompts: {str(e)}")


@app.post('/api/prompts/followup', response_model=PromptsResponse)
async def get_followup_prompts(request: FollowupPromptsRequest):
    """
    Get follow-up prompts that dig deeper into the selected reason.
    Uses conversation history and context for intelligent follow-up questions.
    """
    if gemini_service is None:
        raise HTTPException(status_code=503, detail="Gemini service not configured")
    
    try:
        prompts = gemini_service.generate_followup_prompts(
            emotion=request.emotion,
            selected_option=request.selected_option,
            child_name=request.child_profile.child_name,
            age=request.child_profile.age,
            diagnosis=request.child_profile.diagnosis or "Not specified",
            routine={
                'wake_up_time': request.child_routine.wake_up_time,
                'breakfast_time': request.child_routine.breakfast_time,
                'lunch_time': request.child_routine.lunch_time,
                'snacks_time': request.child_routine.snacks_time,
                'dinner_time': request.child_routine.dinner_time,
                'bedtime': request.child_routine.bedtime,
                'favorite_activities': request.child_routine.favorite_activities,
                'comfort_items': request.child_routine.comfort_items,
                'preferred_prompts': request.child_routine.preferred_prompts
            },
            current_time=request.current_time,
            conversation_history=request.conversation_history
        )
        
        return PromptsResponse(
            session_id=request.session_id,
            prompts=[PromptOption(**p) for p in prompts],
            prompt_type="followup"
        )
    
    except Exception as e:
        print(f"❌ Error generating follow-up prompts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate follow-up prompts: {str(e)}")


@app.post('/api/prompts/dig-deeper', response_model=PromptsResponse)
async def dig_deeper(request: DigDeeperRequest):
    """
    Generate deeper prompts when child wants to explore more.
    Continues the conversation tree to find root cause.
    """
    if gemini_service is None:
        raise HTTPException(status_code=503, detail="Gemini service not configured")
    
    try:
        # Get the last selected option
        last_selection = request.conversation_history[-1] if request.conversation_history else ""
        
        prompts = gemini_service.generate_followup_prompts(
            emotion=request.emotion,
            selected_option=last_selection,
            child_name=request.child_profile.child_name,
            age=request.child_profile.age,
            diagnosis=request.child_profile.diagnosis or "Not specified",
            routine={
                'wake_up_time': request.child_routine.wake_up_time,
                'breakfast_time': request.child_routine.breakfast_time,
                'lunch_time': request.child_routine.lunch_time,
                'snacks_time': request.child_routine.snacks_time,
                'dinner_time': request.child_routine.dinner_time,
                'bedtime': request.child_routine.bedtime,
                'favorite_activities': request.child_routine.favorite_activities,
                'comfort_items': request.child_routine.comfort_items,
                'preferred_prompts': request.child_routine.preferred_prompts
            },
            current_time=request.current_time,
            conversation_history=request.conversation_history[:-1]  # Exclude last one as it's used as selected_option
        )
        
        return PromptsResponse(
            session_id=request.session_id,
            prompts=[PromptOption(**p) for p in prompts],
            prompt_type="dig_deeper"
        )
    
    except Exception as e:
        print(f"❌ Error digging deeper: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to dig deeper: {str(e)}")


@app.post('/api/solution/generate', response_model=SolutionResponse)
async def generate_solution(request: SolutionRequest):
    """
    Generate an empathetic, actionable solution based on the conversation.
    Uses full context to provide personalized recommendations.
    """
    if gemini_service is None:
        raise HTTPException(status_code=503, detail="Gemini service not configured")
    
    try:
        solution = gemini_service.generate_solution(
            emotion=request.emotion,
            conversation_history=request.conversation_history,
            child_name=request.child_profile.child_name,
            age=request.child_profile.age,
            diagnosis=request.child_profile.diagnosis or "Not specified",
            routine={
                'wake_up_time': request.child_routine.wake_up_time,
                'breakfast_time': request.child_routine.breakfast_time,
                'lunch_time': request.child_routine.lunch_time,
                'snacks_time': request.child_routine.snacks_time,
                'dinner_time': request.child_routine.dinner_time,
                'bedtime': request.child_routine.bedtime,
                'favorite_activities': request.child_routine.favorite_activities,
                'comfort_items': request.child_routine.comfort_items,
                'preferred_prompts': request.child_routine.preferred_prompts
            }
        )
        
        return SolutionResponse(
            session_id=request.session_id,
            solution=solution,
            emotion=request.emotion
        )
    
    except Exception as e:
        print(f"❌ Error generating solution: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate solution: {str(e)}")


@app.post('/api/solution/regenerate', response_model=SolutionResponse)
async def regenerate_solution(request: RegenerateSolutionRequest):
    """
    Regenerate a better solution if child is not satisfied.
    Takes previous solution into account to provide an improved version.
    """
    if gemini_service is None:
        raise HTTPException(status_code=503, detail="Gemini service not configured")
    
    try:
        # Add context about previous solution being unsatisfactory
        enhanced_prompt = f"Previous solution was not satisfactory: {request.previous_solution}\n\nGenerate a BETTER, more specific solution."
        
        solution = gemini_service.generate_solution(
            emotion=request.emotion,
            conversation_history=request.conversation_history + [enhanced_prompt],
            child_name=request.child_profile.child_name,
            age=request.child_profile.age,
            diagnosis=request.child_profile.diagnosis or "Not specified",
            routine={
                'wake_up_time': request.child_routine.wake_up_time,
                'breakfast_time': request.child_routine.breakfast_time,
                'lunch_time': request.child_routine.lunch_time,
                'snacks_time': request.child_routine.snacks_time,
                'dinner_time': request.child_routine.dinner_time,
                'bedtime': request.child_routine.bedtime,
                'favorite_activities': request.child_routine.favorite_activities,
                'comfort_items': request.child_routine.comfort_items,
                'preferred_prompts': request.child_routine.preferred_prompts
            }
        )
        
        return SolutionResponse(
            session_id=request.session_id,
            solution=solution,
            emotion=request.emotion
        )
    
    except Exception as e:
        print(f"❌ Error regenerating solution: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to regenerate solution: {str(e)}")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    conversation_history: List[ChatMessage]
    child_profile: ChildProfile
    child_routine: ChildRoutine | None = None


class ChatResponse(BaseModel):
    response: str


@app.post('/api/chat', response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Smart chatbot endpoint for children who can type.
    90% general AI intelligence, 10% personalized based on child profile and routine.
    """
    if gemini_service is None:
        raise HTTPException(status_code=503, detail="Gemini service not configured")
    
    try:
        # Build conversation context
        conversation_context = "\n".join([
            f"{msg.role.capitalize()}: {msg.content}" 
            for msg in request.conversation_history[-10:]  # Last 10 messages for context
        ])
        
        # Build child context (10% personalization) with condition-specific knowledge
        from condition_profiles import get_condition_context
        
        condition_context = get_condition_context(
            request.child_profile.diagnosis or "General",
            request.child_profile.age
        )
        
        child_context = f"""
You are Lumi, a warm, empathetic AI companion chatting with {request.child_profile.child_name}, 
a {request.child_profile.age}-year-old child"""
        
        if request.child_profile.diagnosis:
            child_context += f" with {request.child_profile.diagnosis}"
            
            # Add condition-specific communication style
            if condition_context and condition_context.get('communication_style'):
                child_context += f"\n\nCommunication Style: {condition_context['communication_style']}"
                
                # Add communication preferences
                if condition_context.get('communication_preferences'):
                    prefs = condition_context['communication_preferences'][:2]
                    child_context += f"\nCommunication Tips: {', '.join(prefs)}"
        
        child_context += "."
        
        # Add routine context if available (subtle, not dominant)
        routine_hints = ""
        if request.child_routine:
            current_hour = datetime.now().hour
            if request.child_routine.breakfast_time and 6 <= current_hour < 10:
                routine_hints = " (It's around breakfast time)"
            elif request.child_routine.lunch_time and 11 <= current_hour < 14:
                routine_hints = " (It's around lunch time)"
            elif request.child_routine.dinner_time and 17 <= current_hour < 20:
                routine_hints = " (It's around dinner time)"
            elif request.child_routine.bedtime and 19 <= current_hour < 23:
                routine_hints = " (It's getting close to bedtime)"
        
        # Create the prompt (90% general intelligence, 10% personalization)
        prompt = f"""{child_context}{routine_hints}

You are having a natural, friendly conversation. Be:
- Warm, supportive, and understanding
- Age-appropriate and engaging
- A good listener who asks thoughtful follow-up questions
- Helpful without being preachy
- Encouraging and positive
- Able to discuss any topic the child brings up

Recent conversation:
{conversation_context}

Child's message: {request.message}

Respond naturally as Lumi. Keep responses conversational (2-4 sentences usually). 
Be genuinely interested in what the child is saying. Don't always give advice - sometimes just listen and validate their feelings."""

        # Use Gemini to generate response
        import google.generativeai as genai
        
        # Get next API key
        api_key = gemini_service.get_next_api_key()
        genai.configure(api_key=api_key)
        
        # Use Gemini 2.5 Flash
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        
        return ChatResponse(response=response.text)
    
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")



if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8001)
