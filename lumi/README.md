# Lumi 🌟

An AI-powered assistive communication platform designed to help children with autism identify and resolve emotional challenges through hands-free, accessible interaction.

## Overview

Lumi combines real-time emotion detection, eye-tracking navigation, and conversational AI to provide personalized emotional support for children with autism spectrum disorder. The platform offers a completely hands-free experience, making it accessible for non-verbal children or those with limited motor skills.

## Key Features

### 🎭 Real-Time Emotion Detection
- **EfficientNetV2-based CNN** trained on FER+ dataset
- Detects 7 emotions: happy, sad, angry, surprise, fear, disgust, neutral
- Real-time webcam processing with 95%+ accuracy
- Continuous emotion tracking during 10-second capture window

### 👁️ Hands-Free Eye Tracking Navigation
- **MediaPipe Face Mesh** integration for gaze detection
- Complete hands-free control of all UI elements
- 5-second dwell time for deliberate selection
- Supports:
  - Picture card selection (4 cards)
  - Action buttons (Proceed/Dig Deeper)
  - Solution feedback (This Helps/Try Again)
  - Regenerate options

### 🤖 AI-Powered Conversational Support
- **Google Gemini API** integration for intelligent responses
- Context-aware, multi-turn conversations
- Personalized based on:
  - Child's age and diagnosis
  - Daily routines and preferences
  - Current time and activities
  - Conversation history
- Condition-specific response tailoring (autism, ADHD, anxiety, etc.)

### 📊 Comprehensive User Profiles
- Child profile management with age, diagnosis, and preferences
- Daily routine tracking (meals, activities, comfort items)
- Session history and conversation logs
- Progress tracking and weekly reports

## Tech Stack

### Frontend
- **Framework:** Next.js 14 (App Router)
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **Animations:** Framer Motion
- **Database Client:** Supabase JS

### Backend Services

#### 1. Emotion Detection Service (Port 8000)
- **Framework:** FastAPI
- **Model:** EfficientNetV2 with Attention
- **Computer Vision:** OpenCV, PyTorch
- **Dataset:** FER+ (Facial Expression Recognition Plus)

#### 2. Gemini API Service (Port 8001)
- **Framework:** FastAPI
- **LLM:** Google Gemini 1.5 Flash
- **Features:** Dynamic prompt generation, conversation state management

#### 3. Eye Tracking Service (Port 8002)
- **Framework:** FastAPI + WebSocket
- **Computer Vision:** MediaPipe Face Mesh
- **Real-time Processing:** 30 FPS gaze tracking

### Database
- **Platform:** Supabase (PostgreSQL)
- **Tables:**
  - `child_profiles` - User information and preferences
  - `child_routines` - Daily schedules and activities
  - `sessions` - Conversation sessions and outcomes
  - `session_interactions` - Detailed interaction logs

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Next.js Frontend                         │
│  (TypeScript, Tailwind CSS, Framer Motion)                  │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Emotion    │   │    Gemini    │   │ Eye Tracking │
│   Service    │   │   Service    │   │   Service    │
│  (Port 8000) │   │ (Port 8001)  │   │ (Port 8002)  │
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        │                   ▼                   │
        │          ┌──────────────┐            │
        │          │   Supabase   │            │
        └──────────│  PostgreSQL  │────────────┘
                   └──────────────┘
```

## Installation

### Prerequisites
- Node.js 18+ and npm
- Python 3.9+
- CUDA-capable GPU (optional, for faster emotion detection)
- Webcam

### 1. Clone Repository
```bash
git clone <repository-url>
cd lumi
```

### 2. Frontend Setup
```bash
# Install dependencies
npm install

# Set up environment variables
cp .env.example .env.local
# Edit .env.local with your Supabase credentials
```

### 3. Emotion Detection Service
```bash
cd lumi_backend/emotion_backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained model (if not included)
# Place model at: models/ferplus_checkpoints/best_model.pt
```

### 4. Gemini API Service
```bash
cd lumi_backend/gemini_service

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Google Gemini API key
```

### 5. Eye Tracking Service
```bash
cd lumi_backend/eye_tracking

# Install dependencies (uses same venv as emotion service)
pip install mediapipe opencv-python fastapi uvicorn
```

### 6. Database Setup
```bash
# Run database migrations in Supabase dashboard
# Or use the SQL files in database/migrations/
```

## Running the Application

### Start All Services

**Terminal 1 - Frontend:**
```bash
cd lumi
npm run dev
```

**Terminal 2 - Emotion Detection:**
```bash
cd lumi_backend/emotion_backend
source venv/bin/activate
python -m src.app
```

**Terminal 3 - Gemini Service:**
```bash
cd lumi_backend/gemini_service
source venv/bin/activate
python app.py
```

**Terminal 4 - Eye Tracking:**
```bash
cd lumi_backend/eye_tracking
python eye_tracking_service.py
```

### Access the Application
- Frontend: http://localhost:3000
- Emotion API: http://localhost:8000
- Gemini API: http://localhost:8001
- Eye Tracking: http://localhost:8002

## Usage

### First Time Setup
1. Navigate to http://localhost:3000
2. Complete onboarding:
   - Enter child's name, age, and diagnosis
   - Set up daily routine (meal times, activities)
   - Add comfort items and preferences
3. Enable eye tracking in the menu (optional)

### Using Lumi
1. **Start Session:** Click the "Start" button
2. **Emotion Detection:** Look at the camera for 10 seconds
3. **Problem Identification:** Select a problem card (or use eye tracking)
4. **Conversation:** Follow the guided conversation flow
5. **Solution:** Receive personalized intervention strategies
6. **Feedback:** Indicate if the solution helps

### Eye Tracking Mode
1. Enable eye tracking from the hamburger menu
2. Look at any UI element for 5 seconds to select
3. Progress bar shows selection progress
4. Automatic selection when complete

## Configuration

### Environment Variables

**Frontend (.env.local):**
```env
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_key
```

**Gemini Service (.env):**
```env
GEMINI_API_KEY=your_gemini_api_key
```

**Emotion Service:**
```env
MODEL_PATH=models/ferplus_checkpoints/best_model.pt
USE_CUDA=1  # Set to 0 to force CPU
```

## Project Structure

```
lumi/
├── app/                          # Next.js app directory
│   ├── components/              # React components
│   ├── hooks/                   # Custom React hooks
│   ├── onboarding/             # Onboarding flow
│   ├── start/                  # Main interaction page
│   ├── chat/                   # Text chat interface
│   └── routine/                # Routine setup
├── lib/                         # Utility functions
├── lumi_backend/
│   ├── emotion_backend/        # Emotion detection service
│   │   ├── src/               # Source code
│   │   └── models/            # Trained models
│   ├── gemini_service/        # Gemini API service
│   │   ├── app.py            # FastAPI application
│   │   ├── gemini_prompts_enhanced.py
│   │   └── condition_profiles.py
│   └── eye_tracking/          # Eye tracking service
│       └── eye_tracking_service.py
└── public/                     # Static assets
```

## API Documentation

### Emotion Detection API
**POST** `/predict-frame`
- Input: Image file (JPEG)
- Output: `{ emotion: string, confidence: number }`

### Gemini Service API
**POST** `/api/prompts/initial`
- Generate initial problem identification prompts

**POST** `/api/prompts/followup`
- Generate follow-up conversation prompts

**POST** `/api/solution/generate`
- Generate personalized solution

### Eye Tracking API
**WebSocket** `/ws/eye-tracking`
- Real-time gaze tracking
- Sends: Video frames
- Receives: Gaze data and selections

## Performance

- **Emotion Detection:** ~30ms per frame (GPU), ~100ms (CPU)
- **Eye Tracking:** 30 FPS gaze tracking
- **Gemini Response:** 1-3 seconds average
- **Database Queries:** <50ms average

## Accessibility Features

- ✅ Completely hands-free operation via eye tracking
- ✅ Large, high-contrast UI elements
- ✅ Visual feedback for all interactions
- ✅ Progress indicators for timed selections
- ✅ Picture-based communication (non-verbal friendly)
- ✅ Customizable to individual needs and routines

## Future Enhancements

- [ ] Multi-language support
- [ ] Voice output for solutions
- [ ] Parent/caregiver dashboard
- [ ] Advanced analytics and insights
- [ ] Mobile app version
- [ ] Offline mode support
- [ ] Custom emotion training
- [ ] Integration with therapy programs

## Troubleshooting

### Emotion Detection Not Working
- Check camera permissions
- Ensure emotion service is running on port 8000
- Verify model file exists at correct path

### Eye Tracking Not Connecting
- Check WebSocket connection (port 8002)
- Ensure camera is not in use by another app
- Try disabling and re-enabling eye tracking

### Gemini API Errors
- Verify API key in .env file
- Check API quota and billing
- Ensure service is running on port 8001

### Database Connection Issues
- Verify Supabase credentials
- Check network connectivity
- Review Supabase dashboard for errors


## Acknowledgments

- **FER+ Dataset** for emotion recognition training
- **Google Gemini** for conversational AI
- **MediaPipe** for face mesh and eye tracking
- **Supabase** for database and authentication
- **EfficientNetV2** architecture for emotion detection

## Contributors

ARYAN NANGARATH
ESHWAR-

---


