# Lumi Communication Service

This service uses Google's Gemini API to generate emotion-based picture prompts for children with neurological disabilities.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Gemini API keys:
   - Get API keys from https://makersuite.google.com/app/apikey
   - Set environment variable with comma-separated keys:
   
```bash
# Windows (CMD)
set GEMINI_API_KEYS=your-key-1,your-key-2,your-key-3

# Windows (PowerShell)
$env:GEMINI_API_KEYS="your-key-1,your-key-2,your-key-3"

# Linux/Mac
export GEMINI_API_KEYS=your-key-1,your-key-2,your-key-3
```

3. Run the service:
```bash
python app.py
```

The service will run on `http://localhost:8001`

## API Endpoints

### POST /api/prompts/initial
Generate initial 4 picture prompts based on detected emotion.

Request:
```json
{
  "emotion": "sad",
  "child_profile_id": "uuid",
  "confidence": 0.87
}
```

Response:
```json
{
  "session_id": "uuid",
  "prompts": [
    {
      "label": "Friends and Family",
      "description": "Something about friends or family members"
    },
    ...
  ],
  "prompt_type": "initial"
}
```

### POST /api/prompts/followup
Generate follow-up prompts based on selected category.

Request:
```json
{
  "session_id": "uuid",
  "emotion": "sad",
  "selected_category": "Friends and Family",
  "child_profile_id": "uuid"
}
```

Response:
```json
{
  "session_id": "uuid",
  "prompts": [
    {
      "label": "Had a fight with sibling",
      "description": "Got into an argument with brother or sister"
    },
    ...
  ],
  "prompt_type": "followup"
}
```

## Multiple API Keys

The service supports multiple Gemini API keys for load balancing. Keys are rotated automatically if one fails.
