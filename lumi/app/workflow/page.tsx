"use client";


import React, { useState } from 'react';
import { Database, Split, Brain, TestTube, Zap, Video, Globe, CheckCircle, ArrowRight, ArrowDown, BookOpen, Code } from 'lucide-react';

export default function EmotionWorkflow() {
  const [activePhase, setActivePhase] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState('workflow');

  const mlTerms = [
    {
      term: "Neural Network",
      simple: "Like a brain made of math",
      detailed: "A network of connected nodes (neurons) that process information layer by layer. Each connection has a weight that determines how much influence one neuron has on another.",
      math: "Output = Activation(Weight × Input + Bias)",
      example: "Image pixels → Hidden layers → Emotion prediction"
    },
    {
      term: "Convolution",
      simple: "A sliding filter that detects patterns",
      detailed: "Slides a small matrix (filter/kernel) over the image, multiplying values and summing them up. Different filters detect different features (edges, curves, textures).",
      math: "Conv(x,y) = Σ Σ Image(x+i, y+j) × Filter(i,j)",
      example: "3×3 filter sliding over 224×224 image = 222×222 output"
    },
    {
      term: "Activation Function",
      simple: "Decides if a neuron should fire",
      detailed: "Adds non-linearity to the network. ReLU is most common: outputs the input if positive, else outputs zero.",
      math: "ReLU(x) = max(0, x)",
      example: "Input: [-2, 3, -1, 5] → ReLU → [0, 3, 0, 5]"
    },
    {
      term: "Backpropagation",
      simple: "Learning from mistakes backwards",
      detailed: "After making a prediction, calculates how wrong each weight was and adjusts them. Works backwards from output to input using chain rule.",
      math: "∂Loss/∂Weight = ∂Loss/∂Output × ∂Output/∂Weight",
      example: "Predicted 'happy' but was 'sad' → adjust weights to predict 'sad' better next time"
    },
    {
      term: "Gradient Descent",
      simple: "Walking downhill to find the best answer",
      detailed: "Optimization algorithm that adjusts weights in the direction that reduces loss. Learning rate controls how big each step is.",
      math: "Weight_new = Weight_old - LearningRate × Gradient",
      example: "Loss = 2.5 → adjust weights → Loss = 2.1 → adjust → Loss = 1.8..."
    },
    {
      term: "Loss Function",
      simple: "Measures how wrong the prediction is",
      detailed: "Cross-Entropy Loss penalizes confident wrong predictions heavily. Lower loss = better model.",
      math: "Loss = -Σ y_true × log(y_pred)",
      example: "True: Happy (1,0,0,0,0,0,0), Pred: (0.1,0.7,...) → High Loss!"
    },
    {
      term: "Epoch",
      simple: "One complete pass through all training data",
      detailed: "Model sees every training image once. Training typically runs for 50+ epochs. Model improves gradually each epoch.",
      math: "1 Epoch = N_images / Batch_size iterations",
      example: "28,000 images ÷ 32 batch = 875 iterations per epoch"
    },
    {
      term: "Batch Size",
      simple: "How many images to process at once",
      detailed: "Processing multiple images together is faster and gives more stable gradient updates. Typical: 16-64 images.",
      math: "Gradient = Average(Gradients of all images in batch)",
      example: "Batch of 32: process 32 images → average their gradients → update weights once"
    },
    {
      term: "Learning Rate",
      simple: "How big of a step to take when learning",
      detailed: "Controls how much to adjust weights. Too high = unstable, too low = slow learning. Typically 0.001-0.0001.",
      math: "Step_size = Learning_rate × Gradient",
      example: "LR=0.001: small careful steps. LR=0.1: big risky jumps"
    },
    {
      term: "Transfer Learning",
      simple: "Using knowledge from one task for another",
      detailed: "Start with a model trained on millions of images (ImageNet), then fine-tune it for emotions. Saves tons of training time!",
      math: "Pretrained_Weights + Fine_tuning = Better_Model",
      example: "ResNet knows what faces look like → we just teach it emotions"
    },
    {
      term: "Overfitting",
      simple: "Memorizing instead of learning",
      detailed: "Model performs great on training data but fails on new data. Like memorizing answers instead of understanding concepts.",
      math: "Train_Acc = 99%, Val_Acc = 60% → Overfitting!",
      example: "Model memorizes specific training faces, can't recognize new faces"
    },
    {
      term: "Dropout",
      simple: "Randomly turning off neurons to prevent memorization",
      detailed: "During training, randomly ignore 50% of neurons. Forces network to learn robust features, prevents overfitting.",
      math: "Each neuron has p=0.5 chance of being dropped",
      example: "Layer has 128 neurons → dropout → only 64 active this iteration"
    },
    {
      term: "Pooling",
      simple: "Shrinking the image while keeping important info",
      detailed: "MaxPooling takes the maximum value in each region. Reduces size, keeps strongest features, adds translation invariance.",
      math: "MaxPool 2×2: [1,3 / 2,4] → max = 4",
      example: "224×224 image → MaxPool → 112×112 (half size, key features kept)"
    },
    {
      term: "Softmax",
      simple: "Converting scores to percentages",
      detailed: "Final layer converts raw scores (logits) to probabilities that sum to 100%. Higher scores get exponentially higher probabilities.",
      math: "Softmax(x_i) = e^x_i / Σ e^x_j",
      example: "Scores [2.1, 4.5, 1.2] → Probs [11%, 85%, 4%]"
    },
    {
      term: "Data Augmentation",
      simple: "Creating variations of training images",
      detailed: "Flip, rotate, zoom, color-shift images. Model sees 'different' versions each epoch, learns to be robust to variations.",
      math: "Effective_Dataset = Original_Size × Augmentations",
      example: "1000 images × 10 augmentations = 10,000 effective training samples"
    }
  ];

  const mathExplanations = [
    {
      title: "Forward Pass (Prediction)",
      description: "How an image becomes a prediction",
      steps: [
        {
          step: "Input Image",
          math: "X ∈ R^(224×224×3)",
          explanation: "RGB image with 224×224 pixels, 3 color channels",
          code: "img = torch.tensor([224, 224, 3])"
        },
        {
          step: "Convolution Layer 1",
          math: "H1 = ReLU(Conv(X, W1) + b1)",
          explanation: "Apply filters to detect edges, patterns. ReLU removes negative values.",
          code: "h1 = F.relu(F.conv2d(x, weight1))"
        },
        {
          step: "Pooling",
          math: "P1 = MaxPool(H1, kernel=2)",
          explanation: "Reduce size by taking max value in each 2×2 region",
          code: "p1 = F.max_pool2d(h1, kernel_size=2)"
        },
        {
          step: "More Conv Layers",
          math: "H2 = ReLU(Conv(P1, W2) + b2) ... H50",
          explanation: "50 layers of convolutions learning increasingly complex features",
          code: "# ResNet has 50 such layers"
        },
        {
          step: "Flatten",
          math: "F = Flatten(H50) ∈ R^2048",
          explanation: "Convert 3D feature maps to 1D vector of 2048 numbers",
          code: "f = h50.view(batch, -1)"
        },
        {
          step: "Fully Connected",
          math: "Logits = W_fc × F + b_fc ∈ R^7",
          explanation: "Final layer produces 7 raw scores (one per emotion)",
          code: "logits = self.fc(f)  # [batch, 7]"
        },
        {
          step: "Softmax",
          math: "P(emotion_i) = e^(logit_i) / Σ e^(logit_j)",
          explanation: "Convert logits to probabilities summing to 1.0",
          code: "probs = F.softmax(logits, dim=1)"
        }
      ]
    },
    {
      title: "Backward Pass (Learning)",
      description: "How the model improves from mistakes",
      steps: [
        {
          step: "Calculate Loss",
          math: "L = -log(P(correct_emotion))",
          explanation: "Measure how wrong the prediction was. Lower = better.",
          code: "loss = F.cross_entropy(pred, target)"
        },
        {
          step: "Compute Gradients",
          math: "∂L/∂W = ∂L/∂Output × ∂Output/∂W",
          explanation: "Chain rule: how much each weight contributed to the error",
          code: "loss.backward()  # PyTorch magic!"
        },
        {
          step: "Update Weights",
          math: "W_new = W_old - η × ∂L/∂W",
          explanation: "Move weights in direction that reduces loss. η = learning rate",
          code: "optimizer.step()  # Updates all weights"
        },
        {
          step: "Repeat",
          math: "Iterate for N epochs until convergence",
          explanation: "Do this for every batch, every epoch until loss stops decreasing",
          code: "for epoch in range(50): train()"
        }
      ]
    },
    {
      title: "Real Example with Numbers",
      description: "Actual calculation for one prediction",
      steps: [
        {
          step: "Input",
          math: "[224, 224, 3] = 150,528 numbers",
          explanation: "Image pixels normalized to [0, 1]",
          code: "# Pixel values after normalization"
        },
        {
          step: "After Conv Layers",
          math: "[7, 7, 2048] = 100,352 features",
          explanation: "ResNet extracts 2048 feature maps of size 7×7",
          code: "# High-level features extracted"
        },
        {
          step: "After Flatten",
          math: "2048 features → Dense vector",
          explanation: "One number for each learned feature",
          code: "features = [0.23, -0.15, 0.89, ...]"
        },
        {
          step: "Logits",
          math: "[2.1, -0.5, 1.3, 4.8, 0.2, -1.1, 0.9]",
          explanation: "Raw scores for [angry, disgust, fear, happy, sad, surprise, neutral]",
          code: "logits = self.fc(features)"
        },
        {
          step: "Softmax Math",
          math: "e^4.8 = 121.5 (happy is highest)",
          explanation: "e^2.1=8.2, e^(-0.5)=0.6, ... Sum=145.3",
          code: "# Exponentiate each logit"
        },
        {
          step: "Final Probabilities",
          math: "P(happy) = 121.5/145.3 = 83.6%",
          explanation: "[5.6%, 0.4%, 3.6%, 83.6%, 1.2%, 0.2%, 2.4%]",
          code: "probs = softmax(logits)"
        },
        {
          step: "Prediction",
          math: "argmax(probs) = 3 → 'happy'",
          explanation: "Choose emotion with highest probability",
          code: "pred = probs.argmax()  # Returns 3"
        }
      ]
    }
  ];

  const phases = [
    {
      id: 1,
      icon: Database,
      title: "Phase 1: Data Collection",
      color: "bg-blue-500",
      borderColor: "border-blue-500",
      description: "Gather raw emotion images",
      details: [
        "Download FER2013 dataset or collect custom images",
        "7 emotion categories: angry, disgust, fear, happy, sad, surprise, neutral",
        "Images organized in folders by emotion name",
        "Typical: 28,000+ training images"
      ],
      files: ["Raw images in data/ folder"],
      input: "Internet/Camera",
      output: "data/raw/ (unsplit images)",
      time: "1-2 hours download"
    },
    {
      id: 2,
      icon: Split,
      title: "Phase 2: Data Preparation",
      color: "bg-green-500",
      borderColor: "border-green-500",
      description: "Split into train/validation sets",
      details: [
        "Run split_dataset.py",
        "85% images → Training set",
        "15% images → Validation set",
        "Maintains class balance (equal emotions per set)",
        "Uses random seed (42) for reproducibility"
      ],
      files: ["split_dataset.py"],
      input: "data/train/ (all images)",
      output: "data/train/ (85%) + data/val/ (15%)",
      time: "2-5 minutes",
      command: "python split_dataset.py"
    },
    {
      id: 3,
      icon: Brain,
      title: "Phase 3: Model Training",
      color: "bg-purple-500",
      borderColor: "border-purple-500",
      description: "Teach the AI to recognize emotions",
      details: [
        "Load ResNet50/EfficientNet (pretrained on ImageNet)",
        "For each epoch (50 total):",
        "  • Show all training images",
        "  • Model predicts emotion",
        "  • Calculate loss (how wrong)",
        "  • Update weights using backpropagation",
        "  • Validate on separate images",
        "Save best model automatically",
        "Early stopping if no improvement"
      ],
      files: ["src/train.py", "src/model.py", "src/dataset.py"],
      input: "data/train/ + data/val/",
      output: "models/checkpoints/best_model.pt (100-200MB)",
      time: "2-8 hours (GPU) / 24-48 hours (CPU)",
      command: "./run_train.ps1  OR  python src/train.py --train_dir data/train --val_dir data/val"
    },
    {
      id: 4,
      icon: TestTube,
      title: "Phase 4: Model Evaluation",
      color: "bg-orange-500",
      borderColor: "border-orange-500",
      description: "Test model accuracy",
      details: [
        "Load trained model (best_model.pt)",
        "Test on unseen test dataset",
        "Calculate metrics:",
        "  • Overall accuracy",
        "  • Per-class precision/recall",
        "  • Confusion matrix (where it gets confused)",
        "Generate classification report",
        "Typical accuracy: 60-75% on FER2013"
      ],
      files: ["test_model.py"],
      input: "models/checkpoints/best_model.pt + data/test/",
      output: "Performance metrics, confusion matrix",
      time: "5-10 minutes",
      command: "python test_model.py"
    },
    {
      id: 5,
      icon: Zap,
      title: "Phase 5: Single Image Inference",
      color: "bg-yellow-500",
      borderColor: "border-yellow-500",
      description: "Predict emotion from one image",
      details: [
        "Load trained model",
        "Load single image",
        "Preprocess (resize, normalize)",
        "Run through model",
        "Get probabilities for all 7 emotions",
        "Return highest probability as prediction",
        "Example: Happy (87.3%)"
      ],
      files: ["src/inference.py"],
      input: "best_model.pt + single image",
      output: "{'emotion': 'happy', 'confidence': 0.873}",
      time: "< 1 second",
      command: "python realtime_inference.py --source image --image_path photo.jpg"
    },
    {
      id: 6,
      icon: Video,
      title: "Phase 6: Real-time Detection",
      color: "bg-red-500",
      borderColor: "border-red-500",
      description: "Live webcam emotion recognition",
      details: [
        "Open webcam/video file",
        "For each frame (30 FPS):",
        "  • Detect faces using Haar Cascade",
        "  • Crop each face",
        "  • Predict emotion",
        "  • Draw bounding box + label",
        "Display live results",
        "Press 'q' to quit, 's' to screenshot"
      ],
      files: ["realtime_inference.py"],
      input: "Webcam stream + best_model.pt",
      output: "Live video with emotion labels",
      time: "Real-time (30 FPS)",
      command: "python realtime_inference.py --source webcam"
    },
    {
      id: 7,
      icon: Globe,
      title: "Phase 7: API Deployment",
      color: "bg-indigo-500",
      borderColor: "border-indigo-500",
      description: "Web service for applications",
      details: [
        "Start FastAPI server on port 8000",
        "Load model once at startup",
        "Endpoints available:",
        "  • GET /health (check status)",
        "  • POST /predict-image (single image)",
        "  • POST /predict-batch (multiple images)",
        "Other apps can now use emotion detection",
        "Returns JSON with predictions"
      ],
      files: ["src/app.py"],
      input: "HTTP requests with images",
      output: "JSON responses with predictions",
      time: "Always running (background service)",
      command: "python -m src.app"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-indigo-900 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-white mb-4">
            🧠 Emotion Recognition Complete Guide
          </h1>
          <p className="text-xl text-gray-300">
            Workflow, Math, ML Terms - Everything You Need
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="flex gap-4 mb-8 justify-center flex-wrap">
          <button
            onClick={() => setActiveTab('workflow')}
            className={`px-6 py-3 rounded-lg font-bold transition-all ${
              activeTab === 'workflow'
                ? 'bg-blue-600 text-white shadow-lg scale-105'
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            📊 Workflow Path
          </button>
          <button
            onClick={() => setActiveTab('math')}
            className={`px-6 py-3 rounded-lg font-bold transition-all ${
              activeTab === 'math'
                ? 'bg-purple-600 text-white shadow-lg scale-105'
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            🔢 Math Behind It
          </button>
          <button
            onClick={() => setActiveTab('terms')}
            className={`px-6 py-3 rounded-lg font-bold transition-all ${
              activeTab === 'terms'
                ? 'bg-green-600 text-white shadow-lg scale-105'
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            📚 ML Terms
          </button>
        </div>

        {/* Workflow Tab */}
        {activeTab === 'workflow' && (
          <div>
            <div className="relative mb-16">
              <div className="absolute left-1/2 transform -translate-x-1/2 h-full w-1 bg-gradient-to-b from-blue-500 via-purple-500 to-indigo-500"></div>
              
              {phases.map((phase, index) => {
                const Icon = phase.icon;
                const isLeft = index % 2 === 0;
                
                return (
                  <div key={phase.id} className="relative mb-8">
                    <div className={`flex items-center ${isLeft ? 'flex-row' : 'flex-row-reverse'}`}>
                      <div className={`w-5/12 ${isLeft ? 'pr-8 text-right' : 'pl-8 text-left'}`}>
                        <div 
                          className={`bg-white rounded-xl shadow-2xl p-6 cursor-pointer transform transition-all hover:scale-105 border-l-4 ${phase.borderColor} ${activePhase === phase.id ? 'ring-4 ring-yellow-400' : ''}`}
                          onClick={() => setActivePhase(activePhase === phase.id ? null : phase.id)}
                        >
                          <div className="flex items-center justify-between mb-3">
                            {isLeft ? (
                              <>
                                <div>
                                  <h3 className="text-xl font-bold text-gray-800">{phase.title}</h3>
                                  <p className="text-sm text-gray-600">{phase.description}</p>
                                </div>
                                <div className={`${phase.color} p-3 rounded-full`}>
                                  <Icon className="text-white" size={24} />
                                </div>
                              </>
                            ) : (
                              <>
                                <div className={`${phase.color} p-3 rounded-full`}>
                                  <Icon className="text-white" size={24} />
                                </div>
                                <div>
                                  <h3 className="text-xl font-bold text-gray-800">{phase.title}</h3>
                                  <p className="text-sm text-gray-600">{phase.description}</p>
                                </div>
                              </>
                            )}
                          </div>
                          
                          <div className="text-sm space-y-2 mt-4">
  <div className="bg-blue-100 p-2 rounded text-gray-800">
    <strong className="text-gray-900">Input:</strong> {phase.input}
  </div>
  <div className="bg-green-100 p-2 rounded text-gray-800">
    <strong className="text-gray-900">Output:</strong> {phase.output}
  </div>
  <div className="bg-purple-100 p-2 rounded text-gray-800">
    <strong className="text-gray-900">Time:</strong> {phase.time}
  </div>
</div>
                          {activePhase === phase.id && (
                            <div className="mt-4 pt-4 border-t border-gray-200 text-left">
                              <h4 className="font-bold text-gray-800 mb-2">Detailed Steps:</h4>
                              <ul className="space-y-1 text-sm text-gray-700">
                                {phase.details.map((detail, idx) => (
                                  <li key={idx} className="flex items-start gap-2">
                                    <CheckCircle size={16} className="text-green-500 mt-0.5 flex-shrink-0" />
                                    <span>{detail}</span>
                                  </li>
                                ))}
                              </ul>
                              
                              <h4 className="font-bold text-gray-800 mb-2 mt-4">Files Used:</h4>
                              <div className="flex flex-wrap gap-2">
                                {phase.files.map((file, idx) => (
                                  <code key={idx} className="bg-gray-800 text-green-400 px-2 py-1 rounded text-xs">
                                    {file}
                                  </code>
                                ))}
                              </div>

                              {phase.command && (
                                <>
                                  <h4 className="font-bold text-gray-800 mb-2 mt-4">Command:</h4>
                                  <code className="block bg-gray-900 text-green-400 p-3 rounded text-sm overflow-x-auto">
                                    {phase.command}
                                  </code>
                                </>
                              )}
                            </div>
                          )}
                        </div>
                      </div>

                      <div className="w-2/12 flex justify-center">
                        <div className={`${phase.color} w-16 h-16 rounded-full flex items-center justify-center text-white font-bold text-xl shadow-lg z-10 border-4 border-white`}>
                          {phase.id}
                        </div>
                      </div>

                      <div className="w-5/12"></div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Quick Reference */}
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl p-8 text-white">
              <h2 className="text-3xl font-bold mb-6">🚀 Quick Start Path</h2>
              <div className="space-y-4">
                <div className="flex items-center gap-4">
                  <div className="bg-white text-blue-600 rounded-full w-8 h-8 flex items-center justify-center font-bold flex-shrink-0">1</div>
                  <code className="bg-black bg-opacity-30 px-4 py-2 rounded flex-1">python split_dataset.py</code>
                </div>
                <div className="flex items-center gap-4">
                  <div className="bg-white text-blue-600 rounded-full w-8 h-8 flex items-center justify-center font-bold flex-shrink-0">2</div>
                  <code className="bg-black bg-opacity-30 px-4 py-2 rounded flex-1">./run_train.ps1</code>
                </div>
                <div className="flex items-center gap-4">
                  <div className="bg-white text-blue-600 rounded-full w-8 h-8 flex items-center justify-center font-bold flex-shrink-0">3</div>
                  <code className="bg-black bg-opacity-30 px-4 py-2 rounded flex-1">python test_model.py</code>
                </div>
                <div className="flex items-center gap-4">
                  <div className="bg-white text-blue-600 rounded-full w-8 h-8 flex items-center justify-center font-bold flex-shrink-0">4</div>
                  <code className="bg-black bg-opacity-30 px-4 py-2 rounded flex-1">python realtime_inference.py --source webcam</code>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Math Tab */}
        {activeTab === 'math' && (
          <div className="space-y-8">
            {mathExplanations.map((section, idx) => (
              <div key={idx} className="bg-white rounded-xl shadow-2xl p-8">
                <h2 className="text-3xl font-bold text-gray-800 mb-4">{section.title}</h2>
                <p className="text-gray-600 mb-6">{section.description}</p>
                
                <div className="space-y-6">
                  {section.steps.map((item, stepIdx) => (
                    <div key={stepIdx} className="border-l-4 border-purple-500 pl-6 py-4 bg-purple-50 rounded-r-lg">
                      <div className="flex items-center gap-3 mb-2">
                        <div className="bg-purple-600 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold text-sm">
                          {stepIdx + 1}
                        </div>
                        <h3 className="text-xl font-bold text-gray-800">{item.step}</h3>
                      </div>
                      
                      <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm mb-3 overflow-x-auto">
                        {item.math}
                      </div>
                      
                      <p className="text-gray-700 mb-2">{item.explanation}</p>
                      
                      <div className="bg-blue-900 text-blue-200 p-3 rounded-lg font-mono text-sm overflow-x-auto">
                        {item.code}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
            
            {/* Visual Flow */}
            <div className="bg-white rounded-xl p-8 shadow-2xl">
              <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">📊 Complete Data Flow</h2>
              <div className="flex flex-col items-center space-y-4">
                <div className="bg-blue-100 border-2 border-blue-500 rounded-lg p-4 w-80 text-center">
                  <strong className="text-lg">Image Input</strong>
                  <div className="text-sm text-gray-600 mt-1">224×224×3 = 150,528 pixels</div>
                </div>
                <ArrowDown className="text-blue-500" size={32} />
                
                <div className="bg-green-100 border-2 border-green-500 rounded-lg p-4 w-80 text-center">
                  <strong className="text-lg">Convolution Layers (50×)</strong>
                  <div className="text-sm text-gray-600 mt-1">Extract features: edges → shapes → faces</div>
                </div>
                <ArrowDown className="text-green-500" size={32} />
                
                <div className="bg-purple-100 border-2 border-purple-500 rounded-lg p-4 w-80 text-center">
                  <strong className="text-lg">Feature Vector</strong>
                  <div className="text-sm text-gray-600 mt-1">2048 learned features</div>
                </div>
                <ArrowDown className="text-purple-500" size={32} />
                
                <div className="bg-orange-100 border-2 border-orange-500 rounded-lg p-4 w-80 text-center">
                  <strong className="text-lg">Fully Connected Layer</strong>
                  <div className="text-sm text-gray-600 mt-1">2048 → 7 logits</div>
                </div>
                <ArrowDown className="text-orange-500" size={32} />
                
                <div className="bg-red-100 border-2 border-red-500 rounded-lg p-4 w-80 text-center">
                  <strong className="text-lg">Softmax</strong>
                  <div className="text-sm text-gray-600 mt-1">Convert to probabilities</div>
                </div>
                <ArrowDown className="text-red-500" size={32} />
                
                <div className="bg-yellow-100 border-2 border-yellow-500 rounded-lg p-4 w-80 text-center">
                  <strong className="text-lg">Final Prediction</strong>
                  <div className="text-sm text-gray-600 mt-1">Happy: 83.6%</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ML Terms Tab */}
        {activeTab === 'terms' && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl p-6 shadow-xl mb-8">
              <h2 className="text-3xl font-bold text-gray-800 mb-4">📚 Machine Learning Dictionary</h2>
              <p className="text-gray-600">Click any term to see simple explanation, detailed description, math formula, and example!</p>
            </div>
            
            <div className="grid md:grid-cols-2 gap-6">
              {mlTerms.map((term, idx) => (
                <div key={idx} className="bg-white rounded-xl shadow-xl overflow-hidden hover:shadow-2xl transition-shadow">
                  <div className="bg-gradient-to-r from-indigo-500 to-purple-600 p-4">
                    <h3 className="text-xl font-bold text-white">{term.term}</h3>
                    <p className="text-indigo-100 text-sm italic">{term.simple}</p>
                  </div>
                  
                  <div className="p-6 space-y-4">
                    <div>
                      <h4 className="font-bold text-gray-800 mb-2 flex items-center gap-2">
                        <BookOpen size={18} className="text-blue-600" />
                        Detailed Explanation:
                      </h4>
                      <p className="text-gray-700 text-sm">{term.detailed}</p>
                    </div>
                    
                    <div>
                      <h4 className="font-bold text-gray-800 mb-2 flex items-center gap-2">
                        <Code size={18} className="text-green-600" />
                        Math Formula:
                      </h4>
                      <div className="bg-gray-900 text-green-400 p-3 rounded-lg font-mono text-sm overflow-x-auto">
                        {term.math}
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="font-bold text-gray-800 mb-2">💡 Example:</h4>
                      <div className="bg-blue-50 border-l-4 border-blue-500 p-3 rounded-r-lg text-sm text-gray-700">
                        {term.example}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Key Formulas Summary */}
            <div className="bg-gradient-to-r from-purple-600 to-indigo-600 rounded-xl p-8 text-white mt-12">
              <h2 className="text-3xl font-bold mb-6">🔑 Core Formulas You Must Know</h2>
              <div className="grid md:grid-cols-2 gap-6">
                <div className="bg-white bg-opacity-10 rounded-lg p-4">
                  <h3 className="font-bold text-xl mb-2">Forward Pass</h3>
                  <code className="text-sm">y = σ(Wx + b)</code>
                  <p className="text-sm mt-2 text-purple-100">Output = Activation(Weights × Input + Bias)</p>
                </div>
                
                <div className="bg-white bg-opacity-10 rounded-lg p-4">
                  <h3 className="font-bold text-xl mb-2">Loss Function</h3>
                  <code className="text-sm">L = -Σ y_true × log(y_pred)</code>
                  <p className="text-sm mt-2 text-purple-100">Cross-Entropy measures prediction error</p>
                </div>
                
                <div className="bg-white bg-opacity-10 rounded-lg p-4">
                  <h3 className="font-bold text-xl mb-2">Gradient Descent</h3>
                  <code className="text-sm">W = W - η × ∂L/∂W</code>
                  <p className="text-sm mt-2 text-purple-100">Update weights to minimize loss</p>
                </div>
                
                <div className="bg-white bg-opacity-10 rounded-lg p-4">
                  <h3 className="font-bold text-xl mb-2">Softmax</h3>
                  <code className="text-sm">P(i) = e^(z_i) / Σ e^(z_j)</code>
                  <p className="text-sm mt-2 text-purple-100">Convert scores to probabilities</p>
                </div>
              </div>
            </div>

            {/* Code-to-Math Mapping */}
            <div className="bg-white rounded-xl p-8 shadow-2xl">
              <h2 className="text-3xl font-bold text-gray-800 mb-6">🔗 Code → Math Translation</h2>
              <p className="text-gray-600 mb-6">Understanding how code relates to mathematical concepts</p>
              
              <div className="space-y-6">
                <div className="border-l-4 border-blue-500 pl-6 py-4">
                  <h3 className="font-bold text-lg text-gray-800 mb-2">PyTorch Code</h3>
                  <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm mb-3">
                    output = model(input)
                  </div>
                  <h3 className="font-bold text-lg text-gray-800 mb-2">Mathematical Equivalent</h3>
                  <div className="bg-blue-900 text-blue-200 p-4 rounded-lg font-mono text-sm">
                    y = f(x; θ) = σ(W₅₀(...σ(W₂(σ(W₁x + b₁)) + b₂)...) + b₅₀)
                  </div>
                  <p className="text-gray-600 mt-2 text-sm">50 nested function compositions!</p>
                </div>

                <div className="border-l-4 border-green-500 pl-6 py-4">
                  <h3 className="font-bold text-lg text-gray-800 mb-2">PyTorch Code</h3>
                  <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm mb-3">
                    loss = F.cross_entropy(pred, target)
                  </div>
                  <h3 className="font-bold text-lg text-gray-800 mb-2">Mathematical Equivalent</h3>
                  <div className="bg-blue-900 text-blue-200 p-4 rounded-lg font-mono text-sm">
                    L = -(1/N) Σᵢ Σⱼ yᵢⱼ log(ŷᵢⱼ)
                  </div>
                  <p className="text-gray-600 mt-2 text-sm">Average negative log probability of correct class</p>
                </div>

                <div className="border-l-4 border-purple-500 pl-6 py-4">
                  <h3 className="font-bold text-lg text-gray-800 mb-2">PyTorch Code</h3>
                  <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm mb-3">
                    loss.backward()<br/>
                    optimizer.step()
                  </div>
                  <h3 className="font-bold text-lg text-gray-800 mb-2">Mathematical Equivalent</h3>
                  <div className="bg-blue-900 text-blue-200 p-4 rounded-lg font-mono text-sm">
                    ∂L/∂θ = ∂L/∂y × ∂y/∂θ  (chain rule)<br/>
                    θₙₑw = θₒₗd - η × ∂L/∂θ
                  </div>
                  <p className="text-gray-600 mt-2 text-sm">Automatic differentiation + gradient descent</p>
                </div>

                <div className="border-l-4 border-orange-500 pl-6 py-4">
                  <h3 className="font-bold text-lg text-gray-800 mb-2">PyTorch Code</h3>
                  <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm mb-3">
                    probs = F.softmax(logits, dim=1)
                  </div>
                  <h3 className="font-bold text-lg text-gray-800 mb-2">Mathematical Equivalent</h3>
                  <div className="bg-blue-900 text-blue-200 p-4 rounded-lg font-mono text-sm">
                    P(class=i) = exp(zᵢ) / Σⱼ exp(zⱼ)
                  </div>
                  <p className="text-gray-600 mt-2 text-sm">Normalize logits to probability distribution</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Summary Cards - Always Visible */}
        <div className="grid md:grid-cols-3 gap-6 mt-12">
          <div className="bg-white rounded-xl p-6 shadow-xl">
            <h3 className="text-xl font-bold text-gray-800 mb-3">⏱️ Total Time</h3>
            <p className="text-gray-600">
              <strong>First time setup:</strong> 4-10 hours<br/>
              <strong>Subsequent use:</strong> &lt; 1 second per image
            </p>
          </div>

          <div className="bg-white rounded-xl p-6 shadow-xl">
            <h3 className="text-xl font-bold text-gray-800 mb-3">💾 Storage Needed</h3>
            <p className="text-gray-600">
              <strong>Dataset:</strong> ~500MB<br/>
              <strong>Model:</strong> ~200MB<br/>
              <strong>Dependencies:</strong> ~2GB
            </p>
          </div>

          <div className="bg-white rounded-xl p-6 shadow-xl">
            <h3 className="text-xl font-bold text-gray-800 mb-3">🎯 Typical Accuracy</h3>
            <p className="text-gray-600">
              <strong>FER2013 benchmark:</strong> 60-75%<br/>
              <strong>Custom dataset:</strong> 75-90%<br/>
              <strong>Real-time FPS:</strong> 25-30
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-12 text-white">
          <p className="text-lg opacity-80">
            💡 Navigate between tabs to see workflow, mathematics, and ML terminology!
          </p>
        </div>
      </div>
    </div>
  );
}