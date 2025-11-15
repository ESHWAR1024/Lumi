"""
Condition-Specific Profiles for Enhanced AI Responses
Provides specialized knowledge for each of the 5 supported conditions
"""

CONDITION_PROFILES = {
    "Non-verbal Autism": {
        "communication_style": "Visual and sensory-focused",
        "common_triggers": [
            "Sensory overload (loud noises, bright lights)",
            "Changes in routine",
            "Social situations",
            "Unexpected transitions",
            "Certain textures or foods"
        ],
        "calming_strategies": [
            "Quiet, dimly lit space",
            "Weighted blankets or pressure",
            "Favorite repetitive activities",
            "Visual schedules",
            "Sensory toys (fidgets, stress balls)"
        ],
        "communication_preferences": [
            "Picture cards and visual aids",
            "Simple, concrete language",
            "Extra processing time",
            "Consistent routines",
            "Non-verbal cues (pointing, gestures)"
        ],
        "emotion_patterns": {
            "happy": ["Stimming behaviors", "Repetitive movements", "Vocalizations"],
            "sad": ["Withdrawal", "Reduced engagement", "Self-soothing behaviors"],
            "angry": ["Meltdowns", "Self-injury", "Aggression", "Sensory seeking"],
            "anxious": ["Increased stimming", "Avoidance", "Rigidity"]
        },
        "age_considerations": {
            "5-8": "Focus on basic needs, sensory comfort, and simple choices",
            "9-12": "Include more complex emotions, social situations, and self-regulation",
            "13+": "Address independence, social relationships, and self-advocacy"
        }
    },
    
    "Cerebral Palsy": {
        "communication_style": "May vary from verbal to non-verbal depending on severity",
        "common_triggers": [
            "Physical discomfort or pain",
            "Fatigue from physical effort",
            "Frustration with mobility limitations",
            "Difficulty being understood",
            "Muscle spasms or stiffness"
        ],
        "calming_strategies": [
            "Comfortable positioning",
            "Gentle stretching or massage",
            "Pain management techniques",
            "Assistive devices for comfort",
            "Rest breaks"
        ],
        "communication_preferences": [
            "Allow extra time for responses",
            "Use assistive communication devices",
            "Yes/no questions when needed",
            "Eye gaze or pointing",
            "Patience with speech difficulties"
        ],
        "emotion_patterns": {
            "happy": ["Smiling", "Vocalizations", "Increased movement"],
            "sad": ["Reduced engagement", "Crying", "Withdrawal"],
            "angry": ["Frustration with physical limitations", "Crying", "Tension"],
            "pain": ["Grimacing", "Crying", "Muscle tension", "Reduced movement"]
        },
        "age_considerations": {
            "5-8": "Focus on physical comfort, basic needs, and play",
            "9-12": "Address independence, peer relationships, and self-care",
            "13+": "Include autonomy, social inclusion, and future planning"
        }
    },
    
    "Rett Syndrome": {
        "communication_style": "Primarily non-verbal with eye gaze and limited hand use",
        "common_triggers": [
            "Loss of hand function",
            "Breathing irregularities",
            "Seizures",
            "Gastrointestinal issues",
            "Inability to express needs"
        ],
        "calming_strategies": [
            "Music therapy",
            "Gentle rocking or swinging",
            "Familiar routines",
            "Comfortable positioning",
            "Presence of loved ones"
        ],
        "communication_preferences": [
            "Eye gaze communication",
            "Yes/no questions",
            "Observing facial expressions",
            "Responding to eye contact",
            "Music and rhythm"
        ],
        "emotion_patterns": {
            "happy": ["Eye contact", "Smiling", "Vocalizations", "Engagement"],
            "sad": ["Crying", "Withdrawal", "Reduced eye contact"],
            "distressed": ["Hand wringing", "Breathing changes", "Crying"],
            "content": ["Calm breathing", "Eye engagement", "Relaxed posture"]
        },
        "age_considerations": {
            "5-8": "Focus on comfort, sensory experiences, and connection",
            "9-12": "Maintain engagement, social connection, and joy",
            "13+": "Preserve dignity, choice, and quality of life"
        }
    },
    
    "Childhood Epileptic Encephalopathy": {
        "communication_style": "Varies widely; may be verbal or non-verbal",
        "common_triggers": [
            "Seizure activity or post-ictal state",
            "Medication side effects",
            "Cognitive challenges",
            "Fatigue",
            "Sensory sensitivities"
        ],
        "calming_strategies": [
            "Safe, quiet environment",
            "Consistent routines",
            "Seizure safety measures",
            "Rest and recovery time",
            "Medication management"
        ],
        "communication_preferences": [
            "Simple, clear language",
            "Visual supports",
            "Extra processing time",
            "Patience with cognitive delays",
            "Repetition when needed"
        ],
        "emotion_patterns": {
            "happy": ["Engagement", "Smiling", "Participation"],
            "sad": ["Withdrawal", "Reduced responsiveness", "Crying"],
            "confused": ["Post-seizure disorientation", "Frustration"],
            "tired": ["Reduced engagement", "Irritability", "Need for rest"]
        },
        "age_considerations": {
            "5-8": "Focus on safety, comfort, and simple communication",
            "9-12": "Address learning challenges, social needs, and independence",
            "13+": "Include self-management, social relationships, and future planning"
        }
    },
    
    "ADHD": {
        "communication_style": "Verbal but may have attention and impulse control challenges",
        "common_triggers": [
            "Boredom or lack of stimulation",
            "Overstimulation",
            "Transitions",
            "Frustration with tasks",
            "Social conflicts"
        ],
        "calming_strategies": [
            "Physical movement breaks",
            "Fidget tools",
            "Structured routines",
            "Clear expectations",
            "Positive reinforcement"
        ],
        "communication_preferences": [
            "Brief, clear instructions",
            "Visual reminders",
            "Active listening",
            "Movement while talking",
            "Immediate feedback"
        ],
        "emotion_patterns": {
            "happy": ["High energy", "Enthusiasm", "Rapid speech"],
            "sad": ["Withdrawal", "Low energy", "Negative self-talk"],
            "angry": ["Impulsive reactions", "Frustration", "Difficulty calming"],
            "anxious": ["Restlessness", "Difficulty focusing", "Worry"]
        },
        "age_considerations": {
            "5-8": "Focus on basic self-regulation, routines, and positive behavior",
            "9-12": "Address social skills, organization, and emotional regulation",
            "13+": "Include self-advocacy, time management, and independence"
        }
    }
}


def get_condition_context(diagnosis: str, age: int) -> dict:
    """
    Get condition-specific context for AI prompts
    """
    profile = CONDITION_PROFILES.get(diagnosis, {})
    
    if not profile:
        return {}
    
    # Determine age group
    if age <= 8:
        age_group = "5-8"
    elif age <= 12:
        age_group = "9-12"
    else:
        age_group = "13+"
    
    return {
        "communication_style": profile.get("communication_style", ""),
        "common_triggers": profile.get("common_triggers", []),
        "calming_strategies": profile.get("calming_strategies", []),
        "communication_preferences": profile.get("communication_preferences", []),
        "emotion_patterns": profile.get("emotion_patterns", {}),
        "age_specific_guidance": profile.get("age_considerations", {}).get(age_group, "")
    }


def get_condition_specific_prompt_guidance(diagnosis: str, emotion: str, age: int) -> str:
    """
    Generate condition-specific guidance for prompt generation
    """
    context = get_condition_context(diagnosis, age)
    
    if not context:
        return ""
    
    guidance = f"""
CONDITION-SPECIFIC CONTEXT FOR {diagnosis}:

Communication Style: {context['communication_style']}

Common Triggers for {emotion} emotion:
{chr(10).join(f"- {trigger}" for trigger in context['common_triggers'][:3])}

Effective Calming Strategies:
{chr(10).join(f"- {strategy}" for strategy in context['calming_strategies'][:3])}

Age-Specific Guidance ({age} years old):
{context['age_specific_guidance']}

When generating prompts, consider these condition-specific factors to make suggestions more relevant and helpful.
"""
    
    return guidance
