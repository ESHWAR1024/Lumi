"use client";
import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { supabase, ChildProfile } from "@/lib/supabase";
import { useRouter } from "next/navigation";

const EMOTION_EMOJIS: { [key: string]: string } = {
  happy: "😊",
  sad: "😢",
  angry: "😠",
  surprise: "😮",
  fear: "😨",
  disgust: "🤢",
  neutral: "😐",
};

const EMOTION_COLORS: { [key: string]: string } = {
  happy: "#FFD700",
  sad: "#4169E1",
  angry: "#FF4500",
  surprise: "#FF69B4",
  fear: "#9370DB",
  disgust: "#32CD32",
  neutral: "#808080",
};

export default function StartPage() {
  const router = useRouter();
  const [menuOpen, setMenuOpen] = useState(false);
  const [childProfile, setChildProfile] = useState<ChildProfile | null>(null);
  const [loading, setLoading] = useState(true);
  
  // Emotion recognition states
  const [isRecording, setIsRecording] = useState(false);
  const [emotion, setEmotion] = useState<string>("");
  const [confidence, setConfidence] = useState<number>(0);
  const [allEmotions, setAllEmotions] = useState<{ [key: string]: number }>({});
  const [countdown, setCountdown] = useState<number>(6);
  const [error, setError] = useState<string>("");
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const countdownRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    loadProfile();
  }, []);

  const loadProfile = async () => {
    const storedProfileId = localStorage.getItem("childProfileId");
    if (!storedProfileId) {
      router.push("/onboarding");
      return;
    }

    const { data: routineData } = await supabase
      .from("child_routines")
      .select("*")
      .eq("child_profile_id", storedProfileId)
      .single();

    if (!routineData) {
      router.push("/routine");
      return;
    }

    const { data, error } = await supabase
      .from("child_profiles")
      .select("*")
      .eq("id", storedProfileId)
      .single();

    if (data && !error) {
      setChildProfile(data);
    } else {
      router.push("/onboarding");
    }
    setLoading(false);
  };

  const startCamera = async () => {
    try {
      setError("");
      setEmotion("");
      setConfidence(0);
      setCountdown(45);

      // Request camera access with better constraints
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user" 
        },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        
        // Wait for video to be ready before showing UI
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play().catch(err => {
            console.error("Video play error:", err);
            setError("Failed to start video playback");
          });
        };
      }

      setIsRecording(true);

      // Start emotion detection after 1 second delay (let camera initialize)
      setTimeout(() => {
        intervalRef.current = setInterval(() => {
          captureAndPredict();
        }, 1000); // Predict every second
      }, 1000);

      // Auto-stop after 6 seconds
      countdownRef.current = setInterval(() => {
        setCountdown((prev) => {
          if (prev <= 1) {
            stopCamera();
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    } catch (err: any) {
      const errorMessage = err.name === "NotAllowedError" 
        ? "Camera permission denied. Please allow camera access."
        : err.name === "NotFoundError"
        ? "No camera found on this device."
        : err.name === "NotReadableError"
        ? "Camera is already in use by another application."
        : "Failed to access camera. Please try again.";
      
      setError(errorMessage);
      console.error("Camera error:", err);
    }
  };

  const stopCamera = () => {
    // Stop all tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    // Clear intervals
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (countdownRef.current) {
      clearInterval(countdownRef.current);
      countdownRef.current = null;
    }

    setIsRecording(false);
  };

  const captureAndPredict = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");

    if (!context) return;

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw current video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to blob
    canvas.toBlob(async (blob) => {
      if (!blob) return;

      try {
        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        const response = await fetch("http://localhost:8000/predict-frame", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error("Prediction failed");
        }

        const data = await response.json();

        if (data.success) {
          setEmotion(data.emotion);
          setConfidence(data.confidence);
          setAllEmotions(data.all_probabilities);
        }
      } catch (err) {
        console.error("Prediction error:", err);
        setError("Failed to predict emotion");
      }
    }, "image/jpeg", 0.8);
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen w-full bg-gradient-to-br from-[#FFC8DD] via-[#E0BBE4] to-[#A2D2FF] flex items-center justify-center">
        <p className="text-2xl text-[#2E2E2E]">Loading...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-[#FFC8DD] via-[#E0BBE4] to-[#A2D2FF] flex flex-col items-center justify-center text-[#2E2E2E] relative p-6 overflow-hidden">
      {/* Top Left Hamburger Menu */}
      <div
        className="absolute top-6 left-10 flex flex-col space-y-1 cursor-pointer z-20"
        onClick={() => setMenuOpen(!menuOpen)}
      >
        <div className="w-8 h-1 bg-black rounded-full"></div>
        <div className="w-8 h-1 bg-black rounded-full"></div>
        <div className="w-8 h-1 bg-black rounded-full"></div>
      </div>

      {/* Slide-out Vertical Menu */}
      <motion.div
        initial={{ x: -300, opacity: 0 }}
        animate={menuOpen ? { x: 0, opacity: 1 } : { x: -300, opacity: 0 }}
        transition={{ duration: 0.4, ease: "easeInOut" }}
        className="absolute top-0 left-0 h-full w-72 bg-white/95 backdrop-blur-md text-[#2E2E2E] p-6 pt-20 flex flex-col shadow-2xl z-10"
      >
        {childProfile && (
          <div className="mb-8 pb-6 border-b-2 border-gray-200">
            <h2 className="text-xl font-bold">{childProfile.child_name}</h2>
            <p className="text-sm text-gray-600">{childProfile.age} years old</p>
          </div>
        )}

        <div>
          <h3 className="text-lg font-semibold mb-4 text-gray-700">
            📊 Weekly Reports
          </h3>
          <div className="space-y-3">
            <motion.div
              whileHover={{ scale: 1.02, x: 5 }}
              className="bg-gradient-to-r from-[#A2D2FF]/20 to-[#FFC8DD]/20 p-3 rounded-lg cursor-pointer"
            >
              <p className="font-medium">Week 1 Report</p>
              <p className="text-xs text-gray-600">Latest</p>
            </motion.div>
            <motion.div
              whileHover={{ scale: 1.02, x: 5 }}
              className="bg-gray-100 p-3 rounded-lg cursor-pointer"
            >
              <p className="font-medium">Week 2 Report</p>
              <p className="text-xs text-gray-600">Previous</p>
            </motion.div>
            <motion.div
              whileHover={{ scale: 1.02, x: 5 }}
              className="bg-gray-100 p-3 rounded-lg cursor-pointer"
            >
              <p className="font-medium">Week 3 Report</p>
              <p className="text-xs text-gray-600">Older</p>
            </motion.div>
            <motion.div
              whileHover={{ scale: 1.02, x: 5 }}
              className="bg-gray-100 p-3 rounded-lg cursor-pointer"
            >
              <p className="font-medium">Week 4 Report</p>
              <p className="text-xs text-gray-600">Archive</p>
            </motion.div>
          </div>
        </div>
      </motion.div>

      {/* Overlay to close menu */}
      {menuOpen && (
        <div
          className="absolute inset-0 bg-black/20 z-0"
          onClick={() => setMenuOpen(false)}
        />
      )}

      {/* Main Content */}
      <AnimatePresence mode="wait">
        {!isRecording ? (
          <motion.div
            key="start"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -30 }}
            transition={{ duration: 0.6 }}
            className="flex flex-col items-center justify-center"
          >
            <h1 className="text-8xl font-extrabold mb-8 text-black tracking-wide">
              Lumi 🌟
            </h1>
            <p className="text-xl text-gray-800 mb-10 text-center max-w-lg">
              Press the button below to activate your companion
            </p>

            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
              onClick={startCamera}
              className="bg-gradient-to-r from-[#A2D2FF] to-[#FFC8DD] text-[#2E2E2E] font-semibold text-2xl w-32 h-32 rounded-full shadow-xl flex items-center justify-center"
            >
              Start
            </motion.button>

            {error && (
              <p className="text-red-600 mt-4 text-sm">{error}</p>
            )}
          </motion.div>
        ) : (
          <motion.div
            key="recording"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.5 }}
            className="flex flex-col items-center gap-6 max-w-4xl w-full"
          >
            {/* Video Container */}
            <div className="relative bg-white/90 backdrop-blur-md rounded-3xl shadow-2xl overflow-hidden p-4">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full max-w-2xl rounded-2xl transform scale-x-[-1] bg-black"
                style={{ maxHeight: '480px', minHeight: '360px' }}
              />
              
              {/* Countdown Overlay */}
              <div className="absolute top-8 right-8 bg-black/60 text-white px-4 py-2 rounded-full text-lg font-bold">
                {countdown}s
              </div>

              {/* Camera Status Indicator */}
              <div className="absolute top-8 left-8 flex items-center gap-2 bg-green-500/90 text-white px-3 py-2 rounded-full text-sm font-semibold">
                <div className="w-3 h-3 bg-white rounded-full animate-pulse"></div>
                Live
              </div>

              {/* Loading indicator */}
              {!emotion && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/20 pointer-events-none">
                  <div className="text-white text-lg font-semibold bg-black/60 px-4 py-2 rounded-lg">
                    Initializing camera...
                  </div>
                </div>
              )}
            </div>

            {/* Hidden Canvas for capturing frames */}
            <canvas ref={canvasRef} style={{ display: "none" }} />

            {/* Emotion Display */}
            <AnimatePresence mode="wait">
              {emotion && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="bg-white/90 backdrop-blur-md rounded-2xl p-8 shadow-xl w-full max-w-md"
                >
                  <div className="flex items-center justify-center gap-4 mb-6">
                    <span
                      className="text-6xl"
                      style={{
                        filter: `drop-shadow(0 0 20px ${EMOTION_COLORS[emotion.toLowerCase()] || "#808080"})`,
                      }}
                    >
                      {EMOTION_EMOJIS[emotion.toLowerCase()] || "😐"}
                    </span>
                    <div>
                      <h2 className="text-3xl font-bold capitalize">{emotion}</h2>
                      <p className="text-lg text-gray-600">
                        {(confidence * 100).toFixed(1)}% confident
                      </p>
                    </div>
                  </div>

                  {/* All Emotions Bar Chart */}
                  <div className="space-y-2">
                    {Object.entries(allEmotions)
                      .sort(([, a], [, b]) => b - a)
                      .map(([emo, prob]) => (
                        <div key={emo} className="flex items-center gap-2">
                          <span className="text-sm w-20 capitalize">{emo}</span>
                          <div className="flex-1 bg-gray-200 rounded-full h-3 overflow-hidden">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${prob * 100}%` }}
                              transition={{ duration: 0.5 }}
                              className="h-full rounded-full"
                              style={{
                                backgroundColor: EMOTION_COLORS[emo.toLowerCase()] || "#808080",
                              }}
                            />
                          </div>
                          <span className="text-xs text-gray-600 w-12 text-right">
                            {(prob * 100).toFixed(0)}%
                          </span>
                        </div>
                      ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Stop Button */}
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={stopCamera}
              className="bg-gradient-to-r from-red-400 to-red-600 text-white font-semibold text-xl px-8 py-4 rounded-full shadow-lg"
            >
              Stop Camera
            </motion.button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}