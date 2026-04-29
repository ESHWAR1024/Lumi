# Lumi - Emotion Recognition & Communication Platform for Children with Neurological Disabilities

## 📋 Project Overview

**Lumi** is a comprehensive AI-powered emotion recognition and communication platform designed specifically for children with neurological disabilities (autism, cerebral palsy, Rett syndrome, epilepsy, ADHD). It combines real-time facial emotion detection with an intelligent conversation system to help children identify and express their feelings when verbal communication is difficult or impossible.

### Key Features

- 🎭 **Real-time Emotion Detection** - Uses CNN-based deep learning to detect emotions from facial expressions
- 👁️ **Eye Tracking Interface** - Allows children to select options using only eye movement (for non-verbal users)
- 🗣️ **Intelligent Conversation Flow** - Gemini AI generates context-aware follow-up prompts to identify root causes
- 👨‍👩‍👧 **Personalized Solutions** - Creates tailored interventions based on child profile, routine, and preferences
- ♿ **Accessibility-First Design** - Multiple input methods (click, eye tracking, voice) for varying abilities
- 📊 **Session Tracking** - Records interaction history and generates insights

---

## 🏗️ Architecture

### Frontend
- **Framework**: Next.js 14 + React
- **Styling**: Tailwind CSS + custom animations (GSAP)
- **Eye Tracking**: WebSocket connection to real-time gaze detection service
- **State Management**: React hooks + Supabase for real-time updates

### Backend Services

#### 1. **Emotion Recognition Service** (`lumi_backend/emotion_backend/`)
- **Technology**: PyTorch + FastAPI
- **Models**: EfficientNetV2, ResNet50, ConvNeXt
- **Input**: Webcam feed
- **Output**: Emotion classification (7-8 classes) + confidence scores
- **Port**: 8000
- **Key Files**:
  - `src/model_advanced.py` - Model architectures
  - `src/train_advanced.py` - Training pipeline with mixed precision & augmentation
  - `src/optimal_emotion_model.py` - Right-sized models for different dataset sizes

#### 2. **Eye Tracking Service** (`lumi_backend/eye_tracking/`)
- **Technology**: MediaPipe Face Mesh + FastAPI
- **Purpose**: Real-time gaze detection for hands-free selection
- **Port**: 8002
- **Capabilities**:
  - Picture card selection (4 cards)
  - Action button selection (Proceed/Dig Deeper)
  - Solution feedback buttons (This Helps/Try Again)
  - Regenerate button for alternative problems
- **Key File**: `eye_tracking_service.py`
- **Dwell Time**: 5 seconds per selection

#### 3. **Communication/Prompt Service** (`lumi_backend/gemini_service/`)
- **Technology**: Google Gemini API + FastAPI
- **Purpose**: Generate emotion-based picture prompts and personalized solutions
- **Port**: 8001
- **Endpoints**:
  - `/api/prompts/initial` - Generate 4 initial problem cards
  - `/api/prompts/followup` - Generate deeper follow-up prompts
  - `/api/prompts/regenerate-problems` - Alternative initial cards
  - `/api/prompts/dig-deeper` - Explore root cause further
  - `/api/solution/generate` - Create personalized intervention
  - `/api/chat` - Chatbot for typing-able children
- **Key Files**:
  - `gemini_prompts_enhanced.py` - Context-aware prompt generation
  - `condition_profiles.py` - Condition-specific knowledge base
  - `app.py` - FastAPI server

### Database
- **Supabase** (PostgreSQL)
- **Key Tables**:
  - `child_profiles` - Child information, diagnosis, abilities
  - `child_routines` - Daily schedule and preferences
  - `sessions` - Emotion detection session records
  - `session_interactions` - Conversation flow tracking

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Node.js 18+
- PostgreSQL (via Supabase)
- Google Gemini API keys
- GPU with 6GB+ VRAM (recommended)

### Installation

#### 1. Clone & Setup Frontend
```bash
cd lumi
npm install
cp .env.example .env.local
# Update with your Supabase credentials
npm run dev
```

#### 2. Setup Emotion Recognition Service
```bash
cd lumi_backend/emotion_backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Prepare dataset (FER+ or FER2013 format)
python prepare_affectnet_dataset.py  # Or split_dataset.py

# Train model
python src/train_advanced.py \
  --train_dir data/train \
  --val_dir data/val \
  --model efficientnetv2 \
  --epochs 80 \
  --batch_size 48 \
  --use_amp

# Start service
python -m src.app
```

#### 3. Setup Eye Tracking Service
```bash
cd lumi_backend/eye_tracking
pip install -r requirements.txt
python eye_tracking_service.py
```

#### 4. Setup Communication Service
```bash
cd lumi_backend/gemini_service
pip install -r requirements.txt

# Set API keys
export GEMINI_API_KEYS="key1,key2,key3"

python app.py
```

### Environment Variables
```env
# Frontend (.env.local)
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_key

# Emotion Backend
MODEL_PATH=models/checkpoints/best_model.pt
DEVICE=cuda  # or cpu

# Eye Tracking
PORT=8002

# Communication Service
GEMINI_API_KEYS=key1,key2,key3
```

---

## 📊 User Flow

```
1. EMOTION DETECTION
   ├─ Webcam captures face
   ├─ CNN identifies emotion
   └─ Confidence score determines confidence

2. INITIAL PROBLEM CARDS
   ├─ 4 Fixed emotion-specific cards appear
   └─ Child selects via click or eye gaze

3. PROGRESSIVE NARROWING
   ├─ AI generates 4 specific follow-up cards
   ├─ Each selection narrows down the cause
   └─ Repeat until root cause identified

4. SOLUTION GENERATION
   ├─ AI generates personalized intervention
   ├─ Uses child profile + routine + preferences
   └─ Caregiver acts on solution immediately

5. FEEDBACK
   ├─ Child rates: "This Helps" or "Try Again"
   └─ Session tracked for insights
```

---

## 🧠 AI Architecture

### Emotion Recognition Pipeline
```python
Video Feed
    ↓
Face Detection (Haar Cascade)
    ↓
Face Crop + Preprocessing
    ↓
EfficientNetV2 Backbone
    ↓
Attention Blocks (SE-Block)
    ↓
MLP Classifier
    ↓
Softmax → Emotion Label + Confidence
```

### Conversation Tree Logic
```
INITIAL (Fixed)
├─ Hungry?
├─ Tired?
├─ Missing Someone?
└─ Want to Play?
    │
    FOLLOWUP (AI-Generated)
    ├─ Want Snack?
    ├─ Want Meal?
    └─ Thirsty?
        │
        DEEPER (AI-Generated)
        ├─ Want Cookies?
        ├─ Want Fruit?
        └─ Want Crackers?
            │
            SOLUTION (Personalized)
            └─ "Let's get you cookies + milk"
```

---

## 🔧 Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Next.js 14, React 18 | Web interface |
| **Emotion Detection** | PyTorch, EfficientNetV2 | Facial expression recognition |
| **Eye Tracking** | MediaPipe, WebSocket | Hands-free selection |
| **AI Prompts** | Google Gemini 2.5 Flash | Context-aware conversation |
| **Database** | Supabase/PostgreSQL | Session & profile storage |
| **Styling** | Tailwind CSS, GSAP | UI animations |
| **Deployment** | Docker (optional) | Container orchestration |

---

## 📱 Supported Conditions

Lumi is designed for children with:
- **Non-verbal Autism Spectrum Disorder**
- **Cerebral Palsy**
- **Rett Syndrome**
- **Childhood Epileptic Encephalopathy**
- **ADHD**

Each condition has specialized communication strategies and intervention types.

---

## ⚙️ Configuration

### Model Selection (Emotion Recognition)

Choose based on your dataset size:

```bash
# Small dataset (<20k images)
python train_optimal_model.py --dataset_size small

# Medium dataset (20-40k images) ⭐ RECOMMENDED
python train_optimal_model.py --dataset_size medium

# Large dataset (>40k images)
python train_optimal_model.py --dataset_size large

# Advanced EfficientNetV2
python train_advanced.py --model efficientnetv2 --epochs 80
```

### Eye Tracking Calibration

Adjust in `eye_tracking_service.py`:
```python
SENSITIVITY_X = 22.0        # Horizontal sensitivity
SENSITIVITY_Y = 20.0        # Vertical sensitivity
DWELL_TIME = 5.0            # Seconds to select
HYSTERESIS = 0.08           # Boundary stickiness
```

### AI Prompt Tuning

Modify in `gemini_prompts_enhanced.py`:
- **Temperature**: 0.7 (creativity)
- **Top-P**: 0.95 (diversity)
- **Max Tokens**: 500 (response length)

---

## 📈 Performance Metrics

### Emotion Recognition
- **Accuracy**: 75-82% (validation)
- **Inference Speed**: 20-30 ms per frame
- **GPU Memory**: 2-4 GB
- **CPU Memory**: 1-2 GB

### Eye Tracking
- **Accuracy**: 95% in optimal lighting
- **Latency**: <100ms
- **Frame Rate**: ~30 FPS
- **Dwell Detection**: 5 second dwell time

### Conversation AI
- **Response Time**: 1-2 seconds
- **Prompt Quality**: Contextually relevant 95%+ of time
- **API Cost**: ~$0.001-0.005 per session

---

## 🔒 Privacy & Security

- ✅ All conversations stored locally in Supabase
- ✅ Emotion data encrypted at rest
- ✅ No third-party tracking (except required APIs)
- ✅ GDPR-compliant data retention policies
- ✅ Optional data deletion after session end

---

## 🧪 Testing

```bash
# Test emotion recognition
cd lumi_backend/emotion_backend
python test_model.py --test_dir data/test --model_path models/best_model.pt

# Test eye tracking
cd lumi_backend/eye_tracking
python -m pytest tests/

# Test API endpoints
cd lumi_backend/gemini_service
pytest tests/test_prompts.py
```

---

## 📚 Documentation

- **Emotion Detection**: See `lumi_backend/emotion_backend/README.md`
- **Eye Tracking**: See `lumi/EYE_TRACKING_COMPLETE.md`
- **Conversation Logic**: See `lumi/CONVERSATION_HISTORY_EXPLAINED.md`
- **Database**: See `lumi/DATABASE_UPDATE_GUIDE.md`

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🎯 Future Roadmap

- [ ] Multi-language support (Spanish, Mandarin, ASL)
- [ ] Advanced blink detection for selection
- [ ] Voice input for typing-able children
- [ ] Caregiver dashboard for analytics
- [ ] Mobile app (iOS/Android)
- [ ] Integration with AAC devices
- [ ] Real-time collaboration features
- [ ] Offline mode support

---

## 👥 Authors & Contributors

- **Created for**: Children with neurological disabilities and their caregivers
- **Supported by**: Accessibility-first design principles

---

## 📞 Support & Contact

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: support@lumi-app.com

---

## 🙏 Acknowledgments

- Google Gemini team for AI capabilities
- MediaPipe for eye tracking
- Supabase for backend infrastructure
- The accessibility community for guidance

---

**Made with ❤️ for children with communication challenges.**
