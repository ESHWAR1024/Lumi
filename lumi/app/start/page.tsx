"use client";
import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { supabase, ChildProfile } from "@/lib/supabase";
import { useRouter } from "next/navigation";
import PictureCards from "../components/PictureCards";
import ActionButtons from "../components/ActionButtons";
import SolutionDisplay from "../components/SolutionDisplay";
import RegenerateButton from "../components/RegenerateButton";
import { useSharedEyeTracking } from "../hooks/useSharedEyeTracking";

const EMOTION_COLORS: { [key: string]: string } = {
  happy: "#FFD700",
  sad: "#4169E1",
  angry: "#FF4500",
  surprise: "#FF69B4",
  fear: "#9370DB",
  disgust: "#32CD32",
  neutral: "#808080",
};

interface PromptOption {
  label: string;
  description: string;
  reasoning?: string;
}

export default function StartPage() {
  const router = useRouter();
  const [menuOpen, setMenuOpen] = useState(false);
  const [childProfile, setChildProfile] = useState<ChildProfile | null>(null);
  const [childRoutine, setChildRoutine] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  
  // Emotion recognition states
  const [isRecording, setIsRecording] = useState(false);
  const [emotion, setEmotion] = useState<string>("");
  const [confidence, setConfidence] = useState<number>(0);
  const [countdown, setCountdown] = useState<number>(10);
  const [error, setError] = useState<string>("");
  
  // Conversation flow states
  const [showCards, setShowCards] = useState(false);
  const [prompts, setPrompts] = useState<PromptOption[]>([]);
  const [sessionId, setSessionId] = useState<string>("");
  const [promptType, setPromptType] = useState<"initial" | "followup">("initial");
  const [loadingPrompts, setLoadingPrompts] = useState(false);
  const [conversationHistory, setConversationHistory] = useState<string[]>([]);
  const [interactionDepth, setInteractionDepth] = useState(1);
  const [previousProblems, setPreviousProblems] = useState<string[]>([]);
  
  // Action and solution states
  const [showActionButtons, setShowActionButtons] = useState(false);
  const [showSolution, setShowSolution] = useState(false);
  const [solution, setSolution] = useState("");
  const [showRegenerateButton, setShowRegenerateButton] = useState(false);
  
  // Eye tracking state
  const [eyeTrackingEnabled, setEyeTrackingEnabled] = useState(false);
  
  // Session active state - tracks if a session has started
  const [sessionActive, setSessionActive] = useState(false);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const countdownRef = useRef<NodeJS.Timeout | null>(null);
  const emotionRef = useRef<string>("");
  const emotionHistoryRef = useRef<string[]>([]); // Track all detected emotions

  useEffect(() => {
    loadProfile();
  }, []);

  useEffect(() => {
    if (isRecording && streamRef.current) {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          if (videoRef.current && streamRef.current) {
            videoRef.current.srcObject = streamRef.current;
            console.log("✅ Stream attached to video element!");
          }
        });
      });
    }
  }, [isRecording]);

  const loadProfile = async () => {
    try {
      const storedProfileId = localStorage.getItem("childProfileId");
      if (!storedProfileId) {
        router.push("/onboarding");
        return;
      }

      const { data: routineData, error: routineError } = await supabase
        .from("child_routines")
        .select("*")
        .eq("child_profile_id", storedProfileId)
        .maybeSingle();

      // Check for actual database errors (not "no rows found")
      if (routineError && routineError.code !== "PGRST116") {
        console.error("Failed to load routine:", routineError);
        setError("Failed to load routine. Please check your connection.");
        setLoading(false);
        return;
      }

      // If no routine found, redirect to routine setup
      if (!routineData) {
        console.log("No routine found, redirecting to routine setup");
        router.push("/routine");
        return;
      }

      setChildRoutine(routineData);

      const { data, error } = await supabase
        .from("child_profiles")
        .select("*")
        .eq("id", storedProfileId)
        .single();

      if (error) {
        console.error("Failed to load profile:", error);
        setError("Failed to load profile. Please try refreshing the page.");
        setLoading(false);
        return;
      }

      if (data) {
        // Check if child can type - if true, redirect to chat interface
        if (data.can_type === true) {
          router.push("/chat");
          return;
        }
        setChildProfile(data);
      } else {
        router.push("/onboarding");
      }
    } catch (err) {
      console.error("Unexpected error loading profile:", err);
      setError("An unexpected error occurred. Please refresh the page.");
    } finally {
      setLoading(false);
    }
  };

  const getCurrentTime = () => {
    return new Date().toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit', 
      hour12: false 
    });
  };

  const startCamera = async () => {
    try {
      setError("");
      setEmotion("");
      setConfidence(0);
      setCountdown(10);
      setShowCards(false);
      setShowActionButtons(false);
      setShowSolution(false);
      setShowRegenerateButton(false);
      setPrompts([]);
      setConversationHistory([]);
      setInteractionDepth(1);
      setPreviousProblems([]);
      emotionRef.current = ""; // Reset ref as well
      emotionHistoryRef.current = []; // Reset emotion history
      setSessionActive(true); // Mark session as active

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user" 
        },
        audio: false,
      });

      console.log("✅ Camera stream obtained");
      streamRef.current = stream;
      setIsRecording(true);

      setTimeout(() => {
        intervalRef.current = setInterval(() => {
          captureAndPredict();
        }, 1000);
      }, 2000);

      countdownRef.current = setInterval(() => {
        setCountdown((prev) => {
          if (prev <= 1) {
            stopCameraAndShowCards();
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    } catch (err: any) {
      console.error("❌ Camera access error:", err);
      
      let errorMessage: string;
      if (err.name === "NotAllowedError") {
        errorMessage = "Camera permission denied. Please allow camera access.";
      } else if (err.name === "NotFoundError") {
        errorMessage = "No camera found on this device.";
      } else if (err.name === "NotReadableError") {
        errorMessage = "Camera is already in use by another application.";
      } else {
        errorMessage = "Failed to access camera. Please try again.";
      }
      
      setError(errorMessage);
    }
  };

  const stopCameraAndShowCards = async () => {
    // Get the most frequently detected emotion from history
    const getMostFrequentEmotion = (emotions: string[]): string => {
      if (emotions.length === 0) return "";
      
      const emotionCounts: { [key: string]: number } = {};
      emotions.forEach(emotion => {
        emotionCounts[emotion] = (emotionCounts[emotion] || 0) + 1;
      });
      
      let maxCount = 0;
      let mostFrequent = "";
      Object.entries(emotionCounts).forEach(([emotion, count]) => {
        if (count > maxCount) {
          maxCount = count;
          mostFrequent = emotion;
        }
      });
      
      console.log("📊 Emotion frequency:", emotionCounts);
      return mostFrequent;
    };
    
    const currentEmotion = getMostFrequentEmotion(emotionHistoryRef.current);
    
    console.log("🛑 Stopping camera and fetching prompts...");
    console.log("Most frequent emotion:", currentEmotion);
    console.log("Emotion history:", emotionHistoryRef.current);
    console.log("Child profile:", childProfile);
    stopCamera();
    
    if (!currentEmotion || currentEmotion.trim() === "") {
      console.log("❌ No emotion detected");
      setError("No emotion detected. Please ensure your face is clearly visible and try again.");
      return;
    }
    
    if (!childProfile) {
      console.log("❌ No child profile loaded");
      setError("Profile not loaded. Please refresh the page and try again.");
      return;
    }
    
    // Update state with the most frequent emotion
    setEmotion(currentEmotion);
    emotionRef.current = currentEmotion;
    
    console.log("✅ Conditions met, fetching prompts...");
    await fetchInitialPrompts(currentEmotion);
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

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

  const fetchInitialPrompts = async (emotionValue?: string) => {
    // Use parameter if provided, otherwise use ref, fallback to state
    const currentEmotion = emotionValue || emotionRef.current || emotion;
    console.log("🎯 Fetching initial prompts for emotion:", currentEmotion);
    
    if (!childProfile) {
      console.error("❌ Cannot fetch prompts - childProfile is null");
      setError("Profile not loaded. Please refresh the page and try again.");
      return;
    }
    
    setLoadingPrompts(true);
    setError("");
    
    try {
      // Convert routine arrays to strings for API compatibility
      const formatRoutineForAPI = (routine: any) => {
        if (!routine) {
          return {
            wake_up_time: null,
            breakfast_time: null,
            lunch_time: null,
            snacks_time: null,
            dinner_time: null,
            bedtime: null,
            favorite_activities: null,
            comfort_items: null,
            preferred_prompts: null,
            communication_preferences: null
          };
        }
        
        return {
          wake_up_time: routine.wake_time || routine.wake_up_time || null,
          breakfast_time: routine.breakfast_time || null,
          lunch_time: routine.lunch_time || null,
          snacks_time: routine.snack_times ? (Array.isArray(routine.snack_times) ? routine.snack_times.join(", ") : routine.snack_times) : null,
          dinner_time: routine.dinner_time || null,
          bedtime: routine.bedtime || null,
          favorite_activities: routine.favorite_activities ? (Array.isArray(routine.favorite_activities) ? routine.favorite_activities.join(", ") : routine.favorite_activities) : null,
          comfort_items: routine.comfort_items ? (Array.isArray(routine.comfort_items) ? routine.comfort_items.join(", ") : routine.comfort_items) : null,
          preferred_prompts: routine.preferred_prompts ? (Array.isArray(routine.preferred_prompts) ? routine.preferred_prompts.join(", ") : routine.preferred_prompts) : null,
          communication_preferences: routine.communication_preferences || null
        };
      };
      
      const requestBody = {
        emotion: currentEmotion,
        child_profile_id: childProfile.id,
        confidence: confidence,
        child_profile: {
          child_name: childProfile.child_name,
          age: childProfile.age,
          diagnosis: childProfile.diagnosis || childProfile.condition
        },
        child_routine: formatRoutineForAPI(childRoutine),
        current_time: getCurrentTime()
      };
      
      console.log("📤 Sending request body:", JSON.stringify(requestBody, null, 2));
      
      const response = await fetch("http://localhost:8001/api/prompts/initial", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        console.error("❌ API Error Response:", errorData);
        throw new Error(`HTTP error! status: ${response.status}${errorData ? `: ${JSON.stringify(errorData)}` : ''}`);
      }

      const data = await response.json();
      console.log("✅ Received prompts:", data);
      
      setPrompts(data.prompts);
      setSessionId(data.session_id);
      setPromptType("initial");
      setShowCards(true);
      setShowRegenerateButton(true); // Show regenerate button after initial prompts load
      
      // Store problem labels in previousProblems
      const problemLabels = data.prompts.map((prompt: PromptOption) => prompt.label);
      setPreviousProblems(problemLabels);
      
      // Create session in database with error handling
      try {
        await createSession(data.session_id);
      } catch (sessionErr) {
        console.error("⚠️ Session creation failed, but continuing with prompts:", sessionErr);
        // Don't block the user experience if session creation fails
        // The prompts are already loaded and displayed
      }
    } catch (err) {
      console.error("❌ Failed to fetch prompts:", err);
      setError("Failed to load picture cards. Is Gemini service running on port 8001?");
    } finally {
      setLoadingPrompts(false);
    }
  };

  const fetchFollowupPrompts = async (selectedOption: string) => {
    if (!childProfile) {
      console.error("❌ Cannot fetch follow-up prompts - childProfile is null");
      setError("Profile not loaded. Please refresh the page and try again.");
      return;
    }
    
    setLoadingPrompts(true);
    setError("");
    
    try {
      // Use the same formatting function for consistency
      const formatRoutineForAPI = (routine: any) => {
        if (!routine) {
          return {
            wake_up_time: null,
            breakfast_time: null,
            lunch_time: null,
            snacks_time: null,
            dinner_time: null,
            bedtime: null,
            favorite_activities: null,
            comfort_items: null,
            preferred_prompts: null,
            communication_preferences: null
          };
        }
        
        return {
          wake_up_time: routine.wake_time || routine.wake_up_time || null,
          breakfast_time: routine.breakfast_time || null,
          lunch_time: routine.lunch_time || null,
          snacks_time: routine.snack_times ? (Array.isArray(routine.snack_times) ? routine.snack_times.join(", ") : routine.snack_times) : null,
          dinner_time: routine.dinner_time || null,
          bedtime: routine.bedtime || null,
          favorite_activities: routine.favorite_activities ? (Array.isArray(routine.favorite_activities) ? routine.favorite_activities.join(", ") : routine.favorite_activities) : null,
          comfort_items: routine.comfort_items ? (Array.isArray(routine.comfort_items) ? routine.comfort_items.join(", ") : routine.comfort_items) : null,
          preferred_prompts: routine.preferred_prompts ? (Array.isArray(routine.preferred_prompts) ? routine.preferred_prompts.join(", ") : routine.preferred_prompts) : null,
          communication_preferences: routine.communication_preferences || null
        };
      };
      
      const response = await fetch("http://localhost:8001/api/prompts/followup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          emotion: emotion,
          selected_option: selectedOption,
          child_profile_id: childProfile.id,
          child_profile: {
            child_name: childProfile.child_name,
            age: childProfile.age,
            diagnosis: childProfile.diagnosis || childProfile.condition
          },
          child_routine: formatRoutineForAPI(childRoutine),
          current_time: getCurrentTime(),
          interaction_depth: interactionDepth,
          conversation_history: conversationHistory
        })
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      
      if (!data.prompts || data.prompts.length === 0) {
        throw new Error("No prompts received from server");
      }
      
      setPrompts(data.prompts);
      setPromptType("followup");
      
      await storeInteraction(selectedOption, data.prompts, "followup");
    } catch (err) {
      console.error("Failed to fetch follow-up prompts:", err);
      const errorMessage = err instanceof Error ? err.message : "Unknown error";
      setError(`Failed to load follow-up cards: ${errorMessage}. Please try again.`);
      setShowCards(false);
    } finally {
      setLoadingPrompts(false);
    }
  };

  const handleRegenerateProblems = async () => {
    if (!childProfile) {
      console.error("❌ Cannot regenerate problems - childProfile is null");
      setError("Profile not loaded. Please refresh the page and try again.");
      return;
    }
    
    setLoadingPrompts(true);
    setError("");
    
    try {
      const formatRoutineForAPI = (routine: any) => {
        if (!routine) {
          return {
            wake_up_time: null,
            breakfast_time: null,
            lunch_time: null,
            snacks_time: null,
            dinner_time: null,
            bedtime: null,
            favorite_activities: null,
            comfort_items: null,
            preferred_prompts: null,
            communication_preferences: null
          };
        }
        
        return {
          wake_up_time: routine.wake_time || routine.wake_up_time || null,
          breakfast_time: routine.breakfast_time || null,
          lunch_time: routine.lunch_time || null,
          snacks_time: routine.snack_times ? (Array.isArray(routine.snack_times) ? routine.snack_times.join(", ") : routine.snack_times) : null,
          dinner_time: routine.dinner_time || null,
          bedtime: routine.bedtime || null,
          favorite_activities: routine.favorite_activities ? (Array.isArray(routine.favorite_activities) ? routine.favorite_activities.join(", ") : routine.favorite_activities) : null,
          comfort_items: routine.comfort_items ? (Array.isArray(routine.comfort_items) ? routine.comfort_items.join(", ") : routine.comfort_items) : null,
          preferred_prompts: routine.preferred_prompts ? (Array.isArray(routine.preferred_prompts) ? routine.preferred_prompts.join(", ") : routine.preferred_prompts) : null,
          communication_preferences: routine.communication_preferences || null
        };
      };
      
      const requestBody = {
        session_id: sessionId,
        emotion: emotion,
        previous_problems: previousProblems,
        child_profile: {
          child_name: childProfile.child_name,
          age: childProfile.age,
          diagnosis: childProfile.diagnosis || childProfile.condition
        },
        child_routine: formatRoutineForAPI(childRoutine),
        current_time: getCurrentTime()
      };
      
      console.log("📤 Regenerating problems with request:", JSON.stringify(requestBody, null, 2));
      
      const response = await fetch("http://localhost:8001/api/prompts/regenerate-problems", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        console.error("❌ API Error Response:", errorData);
        throw new Error(`HTTP error! status: ${response.status}${errorData ? `: ${JSON.stringify(errorData)}` : ''}`);
      }

      const data = await response.json();
      console.log("✅ Received regenerated prompts:", data);
      
      // Update prompts state with 4 new cards
      setPrompts(data.prompts);
      
      // Append new problem labels to previousProblems
      const newProblemLabels = data.prompts.map((prompt: PromptOption) => prompt.label);
      setPreviousProblems(prev => [...prev, ...newProblemLabels]);
    } catch (err) {
      console.error("❌ Failed to regenerate problems:", err);
      setError("Failed to generate new problem options. Please try again.");
    } finally {
      setLoadingPrompts(false);
    }
  };

  const handleDigDeeper = async () => {
    if (!childProfile) {
      console.error("❌ Cannot dig deeper - childProfile is null");
      setError("Profile not loaded. Please refresh the page and try again.");
      return;
    }
    
    setShowActionButtons(false);
    setLoadingPrompts(true);
    setError("");
    
    try {
      const formatRoutineForAPI = (routine: any) => {
        if (!routine) {
          return {
            wake_up_time: null,
            breakfast_time: null,
            lunch_time: null,
            snacks_time: null,
            dinner_time: null,
            bedtime: null,
            favorite_activities: null,
            comfort_items: null,
            preferred_prompts: null,
            communication_preferences: null
          };
        }
        
        return {
          wake_up_time: routine.wake_time || routine.wake_up_time || null,
          breakfast_time: routine.breakfast_time || null,
          lunch_time: routine.lunch_time || null,
          snacks_time: routine.snack_times ? (Array.isArray(routine.snack_times) ? routine.snack_times.join(", ") : routine.snack_times) : null,
          dinner_time: routine.dinner_time || null,
          bedtime: routine.bedtime || null,
          favorite_activities: routine.favorite_activities ? (Array.isArray(routine.favorite_activities) ? routine.favorite_activities.join(", ") : routine.favorite_activities) : null,
          comfort_items: routine.comfort_items ? (Array.isArray(routine.comfort_items) ? routine.comfort_items.join(", ") : routine.comfort_items) : null,
          preferred_prompts: routine.preferred_prompts ? (Array.isArray(routine.preferred_prompts) ? routine.preferred_prompts.join(", ") : routine.preferred_prompts) : null,
          communication_preferences: routine.communication_preferences || null
        };
      };
      
      const response = await fetch("http://localhost:8001/api/prompts/dig-deeper", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          emotion: emotion,
          conversation_history: conversationHistory,
          child_profile: {
            child_name: childProfile.child_name,
            age: childProfile.age,
            diagnosis: childProfile.diagnosis || childProfile.condition
          },
          child_routine: formatRoutineForAPI(childRoutine),
          current_time: getCurrentTime()
        })
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      
      if (!data.prompts || data.prompts.length === 0) {
        throw new Error("No prompts received from server");
      }
      
      setPrompts(data.prompts);
      setShowCards(true);
      setInteractionDepth(prev => prev + 1);
      
      await storeInteraction("dig_deeper", data.prompts, "dig_deeper");
    } catch (err) {
      console.error("Failed to dig deeper:", err);
      const errorMessage = err instanceof Error ? err.message : "Unknown error";
      setError(`Failed to generate deeper prompts: ${errorMessage}. Please try again.`);
      setShowActionButtons(true);
    } finally {
      setLoadingPrompts(false);
    }
  };

  const handleProceedToSolution = async () => {
    if (!childProfile) {
      console.error("❌ Cannot proceed to solution - childProfile is null");
      setError("Profile not loaded. Please refresh the page and try again.");
      return;
    }
    
    setShowActionButtons(false);
    setLoadingPrompts(true);
    setError("");
    
    try {
      const formatRoutineForAPI = (routine: any) => {
        if (!routine) {
          return {
            wake_up_time: null,
            breakfast_time: null,
            lunch_time: null,
            snacks_time: null,
            dinner_time: null,
            bedtime: null,
            favorite_activities: null,
            comfort_items: null,
            preferred_prompts: null,
            communication_preferences: null
          };
        }
        
        return {
          wake_up_time: routine.wake_time || routine.wake_up_time || null,
          breakfast_time: routine.breakfast_time || null,
          lunch_time: routine.lunch_time || null,
          snacks_time: routine.snack_times ? (Array.isArray(routine.snack_times) ? routine.snack_times.join(", ") : routine.snack_times) : null,
          dinner_time: routine.dinner_time || null,
          bedtime: routine.bedtime || null,
          favorite_activities: routine.favorite_activities ? (Array.isArray(routine.favorite_activities) ? routine.favorite_activities.join(", ") : routine.favorite_activities) : null,
          comfort_items: routine.comfort_items ? (Array.isArray(routine.comfort_items) ? routine.comfort_items.join(", ") : routine.comfort_items) : null,
          preferred_prompts: routine.preferred_prompts ? (Array.isArray(routine.preferred_prompts) ? routine.preferred_prompts.join(", ") : routine.preferred_prompts) : null,
          communication_preferences: routine.communication_preferences || null
        };
      };
      
      const response = await fetch("http://localhost:8001/api/solution/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          emotion: emotion,
          conversation_history: conversationHistory,
          child_profile: {
            child_name: childProfile.child_name,
            age: childProfile.age,
            diagnosis: childProfile.diagnosis || childProfile.condition
          },
          child_routine: formatRoutineForAPI(childRoutine)
        })
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      
      if (!data.solution) {
        throw new Error("No solution received from server");
      }
      
      setSolution(data.solution);
      setShowSolution(true);
      
      const { error: updateError } = await supabase
        .from("sessions")
        .update({ solution_provided: data.solution })
        .eq("id", sessionId);
      
      if (updateError) {
        console.error("Failed to update session with solution:", updateError);
      }
    } catch (err) {
      console.error("Failed to generate solution:", err);
      const errorMessage = err instanceof Error ? err.message : "Unknown error";
      setError(`Failed to generate solution: ${errorMessage}. Please try again.`);
      setShowActionButtons(true);
    } finally {
      setLoadingPrompts(false);
    }
  };

  const resetSession = () => {
    // Reset emotion and confidence
    setEmotion("");
    setConfidence(0);
    
    // Clear conversation history
    setConversationHistory([]);
    
    // Reset interaction depth to 1
    setInteractionDepth(1);
    
    // Clear prompts and solution
    setPrompts([]);
    setSolution("");
    
    // Clear previousProblems array
    setPreviousProblems([]);
    
    // Hide all UI components (cards, actions, solution, regenerate button)
    setShowCards(false);
    setShowActionButtons(false);
    setShowSolution(false);
    setShowRegenerateButton(false);
    
    // Clear session ID
    setSessionId("");
    
    // Reset prompt type
    setPromptType("initial");
    
    // Clear any errors
    setError("");
    
    // Mark session as inactive - this will show the logo and start button again
    setSessionActive(false);
  };

  const handleSatisfied = async () => {
    if (!childProfile) {
      console.error("❌ Cannot save satisfaction - childProfile is null");
      setError("Profile not loaded. Please refresh the page.");
      return;
    }
    
    try {
      // Create session summary with child name, emotion, conversation path, and solution
      const summary = `${childProfile.child_name} felt ${emotion}. ${conversationHistory.join(" → ")}. Solution: ${solution}`;
      
      // Update session with solution, satisfaction_status as "satisfied"
      // Set session status to "completed" and record ended_at timestamp
      const { error: updateError } = await supabase
        .from("sessions")
        .update({
          solution_provided: solution,
          satisfaction_status: "satisfied",
          session_summary: summary,
          status: "completed",
          ended_at: new Date().toISOString()
        })
        .eq("id", sessionId);
      
      if (updateError) {
        console.error("Failed to update session:", updateError);
        throw new Error(`Database error: ${updateError.message}`);
      }
      
      // Show success message to user
      alert(`Thank you, ${childProfile.child_name}! I'm glad I could help. 😊`);
      
      // Reset all state variables for new session
      resetSession();
    } catch (err) {
      console.error("Failed to save satisfaction:", err);
      const errorMessage = err instanceof Error ? err.message : "Unknown error";
      setError(`Failed to save your feedback: ${errorMessage}. Please try again.`);
    }
  };

  const handleNotSatisfied = async () => {
    if (!childProfile) {
      console.error("❌ Cannot regenerate solution - childProfile is null");
      setError("Profile not loaded. Please refresh the page and try again.");
      return;
    }
    
    setLoadingPrompts(true);
    setError("");
    
    try {
      const formatRoutineForAPI = (routine: any) => {
        if (!routine) {
          return {
            wake_up_time: null,
            breakfast_time: null,
            lunch_time: null,
            snacks_time: null,
            dinner_time: null,
            bedtime: null,
            favorite_activities: null,
            comfort_items: null,
            preferred_prompts: null,
            communication_preferences: null
          };
        }
        
        return {
          wake_up_time: routine.wake_time || routine.wake_up_time || null,
          breakfast_time: routine.breakfast_time || null,
          lunch_time: routine.lunch_time || null,
          snacks_time: routine.snack_times ? (Array.isArray(routine.snack_times) ? routine.snack_times.join(", ") : routine.snack_times) : null,
          dinner_time: routine.dinner_time || null,
          bedtime: routine.bedtime || null,
          favorite_activities: routine.favorite_activities ? (Array.isArray(routine.favorite_activities) ? routine.favorite_activities.join(", ") : routine.favorite_activities) : null,
          comfort_items: routine.comfort_items ? (Array.isArray(routine.comfort_items) ? routine.comfort_items.join(", ") : routine.comfort_items) : null,
          preferred_prompts: routine.preferred_prompts ? (Array.isArray(routine.preferred_prompts) ? routine.preferred_prompts.join(", ") : routine.preferred_prompts) : null,
          communication_preferences: routine.communication_preferences || null
        };
      };
      
      const response = await fetch("http://localhost:8001/api/solution/regenerate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          emotion: emotion,
          conversation_history: conversationHistory,
          previous_solution: solution,
          child_profile: {
            child_name: childProfile.child_name,
            age: childProfile.age,
            diagnosis: childProfile.diagnosis || childProfile.condition
          },
          child_routine: formatRoutineForAPI(childRoutine)
        })
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      
      if (!data.solution) {
        throw new Error("No solution received from server");
      }
      
      setSolution(data.solution);
      
      const { error: updateError } = await supabase
        .from("sessions")
        .update({ solution_provided: data.solution })
        .eq("id", sessionId);
      
      if (updateError) {
        console.error("Failed to update session with new solution:", updateError);
      }
    } catch (err) {
      console.error("Failed to regenerate solution:", err);
      const errorMessage = err instanceof Error ? err.message : "Unknown error";
      setError(`Failed to generate new solution: ${errorMessage}. Please try again.`);
    } finally {
      setLoadingPrompts(false);
    }
  };

  const createSession = async (sid: string) => {
    if (!childProfile) {
      console.error("❌ Cannot create session - childProfile is null");
      throw new Error("Child profile not loaded");
    }
    
    try {
      const { error } = await supabase.from("sessions").insert({
        id: sid,
        child_profile_id: childProfile.id,
        initial_emotion: emotion,
        emotion_confidence: confidence,
        status: "active"
      });
      
      if (error) {
        console.error("❌ Database error creating session:", error);
        throw new Error(`Failed to create session: ${error.message}`);
      }
      
      console.log("✅ Session created successfully:", sid);
    } catch (err) {
      console.error("❌ Failed to create session:", err);
      throw err;
    }
  };

  const storeInteraction = async (selected: string, promptOptions: PromptOption[], type: string) => {
    try {
      const { data: existingInteractions, error: fetchError } = await supabase
        .from("session_interactions")
        .select("interaction_order")
        .eq("session_id", sessionId)
        .order("interaction_order", { ascending: false })
        .limit(1);

      if (fetchError) {
        console.error("Failed to fetch existing interactions:", fetchError);
        throw new Error(`Database error: ${fetchError.message}`);
      }

      const nextOrder = existingInteractions && existingInteractions.length > 0 
        ? existingInteractions[0].interaction_order + 1 
        : 1;

      const { error: insertError } = await supabase.from("session_interactions").insert({
        session_id: sessionId,
        interaction_order: nextOrder,
        prompt_type: type,
        selected_option: selected,
        prompt_options: promptOptions,
        interaction_depth: interactionDepth,
        action_type: type
      });
      
      if (insertError) {
        console.error("Failed to insert interaction:", insertError);
        throw new Error(`Database error: ${insertError.message}`);
      }
      
      console.log("✅ Interaction stored successfully");
    } catch (err) {
      console.error("Failed to store interaction:", err);
      // Don't throw - this is a non-critical operation that shouldn't block the user flow
    }
  };

  const handleCardSelect = async (selectedLabel: string) => {
    setConversationHistory(prev => [...prev, selectedLabel]);
    setShowRegenerateButton(false); // Hide regenerate button after card selection
    
    if (promptType === "initial") {
      setShowCards(false);
      await fetchFollowupPrompts(selectedLabel);
      setShowCards(true);
    } else {
      setShowCards(false);
      setShowActionButtons(true);
    }
  };

  // Eye tracking selection handler
  const handleEyeTrackingSelect = (index: number) => {
    // Handle card selection (0-3)
    if (index >= 0 && index <= 3 && prompts[index]) {
      handleCardSelect(prompts[index].label);
    }
    // Handle action button selection (4-5)
    else if (index === 4) {
      handleProceedToSolution();
    } else if (index === 5) {
      handleDigDeeper();
    }
    // Handle solution button selection (6-7)
    else if (index === 6) {
      handleSatisfied();
    } else if (index === 7) {
      handleNotSatisfied();
    }
    // Handle regenerate button selection (8)
    else if (index === 8) {
      handleRegenerateProblems();
    }
  };

  // Determine current mode based on what's visible
  const eyeTrackingMode = showSolution 
    ? 'solution' 
    : showActionButtons 
    ? 'buttons' 
    : (showCards && showRegenerateButton && promptType === "initial")
    ? 'cards_with_regenerate'
    : 'cards';

  // Single shared eye tracking hook
  const { gazeData, videoRef: eyeVideoRef, canvasRef: eyeCanvasRef, connected: eyeTrackingConnected } = useSharedEyeTracking(
    eyeTrackingEnabled && (showCards || showActionButtons || showSolution),
    handleEyeTrackingSelect,
    eyeTrackingMode,
    'main'
  );

  const handleInitialDigDeeper = async () => {
    if (!childProfile) {
      console.error("❌ Cannot generate alternative prompts - childProfile is null");
      setError("Profile not loaded. Please refresh the page and try again.");
      return;
    }
    
    setShowCards(false);
    setLoadingPrompts(true);
    setError("");
    
    try {
      const formatRoutineForAPI = (routine: any) => {
        if (!routine) {
          return {
            wake_up_time: null,
            breakfast_time: null,
            lunch_time: null,
            snacks_time: null,
            dinner_time: null,
            bedtime: null,
            favorite_activities: null,
            comfort_items: null,
            preferred_prompts: null,
            communication_preferences: null
          };
        }
        
        return {
          wake_up_time: routine.wake_time || routine.wake_up_time || null,
          breakfast_time: routine.breakfast_time || null,
          lunch_time: routine.lunch_time || null,
          snacks_time: routine.snack_times ? (Array.isArray(routine.snack_times) ? routine.snack_times.join(", ") : routine.snack_times) : null,
          dinner_time: routine.dinner_time || null,
          bedtime: routine.bedtime || null,
          favorite_activities: routine.favorite_activities ? (Array.isArray(routine.favorite_activities) ? routine.favorite_activities.join(", ") : routine.favorite_activities) : null,
          comfort_items: routine.comfort_items ? (Array.isArray(routine.comfort_items) ? routine.comfort_items.join(", ") : routine.comfort_items) : null,
          preferred_prompts: routine.preferred_prompts ? (Array.isArray(routine.preferred_prompts) ? routine.preferred_prompts.join(", ") : routine.preferred_prompts) : null,
          communication_preferences: routine.communication_preferences || null
        };
      };
      
      // Use the current emotion from ref
      const currentEmotion = emotionRef.current || emotion;
      
      const response = await fetch("http://localhost:8001/api/prompts/initial", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          emotion: currentEmotion,
          child_profile_id: childProfile.id,
          confidence: confidence,
          child_profile: {
            child_name: childProfile.child_name,
            age: childProfile.age,
            diagnosis: childProfile.diagnosis || childProfile.condition
          },
          child_routine: formatRoutineForAPI(childRoutine),
          current_time: getCurrentTime()
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        console.error("❌ API Error Response:", errorData);
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("✅ Received alternative prompts:", data);
      
      setPrompts(data.prompts);
      setShowCards(true);
      
      // Store this as a "dig_deeper" action on initial prompts
      await storeInteraction("none_of_these", data.prompts, "initial_dig_deeper");
    } catch (err) {
      console.error("Failed to generate alternative prompts:", err);
      const errorMessage = err instanceof Error ? err.message : "Unknown error";
      setError(`Failed to load alternative cards: ${errorMessage}. Please try again.`);
      setShowCards(true); // Show the original cards again
    } finally {
      setLoadingPrompts(false);
    }
  };

  const captureAndPredict = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");

    if (!context) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
      if (!blob) {
        console.error("Failed to create blob from canvas");
        return;
      }

      try {
        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        const response = await fetch("http://localhost:8000/predict-frame", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Emotion detection service error: ${response.status}`);
        }

        const data = await response.json();

        if (data.success && data.emotion && data.confidence !== undefined) {
          console.log("✅ Emotion detected:", data.emotion, "Confidence:", data.confidence);
          setEmotion(data.emotion);
          setConfidence(data.confidence);
          emotionRef.current = data.emotion; // Store in ref for countdown callback
          emotionHistoryRef.current.push(data.emotion); // Track all emotions
        } else {
          console.warn("⚠️ Invalid response from emotion detection service:", data);
        }
      } catch (err) {
        console.error("Prediction error:", err);
        // Don't set error state here as this runs continuously
        // Only log for debugging purposes
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

        {/* Eye Tracking Toggle */}
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-3 text-gray-700">
            👁️ Eye Tracking
          </h3>
          <div className="flex items-center justify-between bg-gradient-to-r from-purple-100 to-pink-100 p-4 rounded-lg">
            <div>
              <p className="font-medium text-sm">Hands-Free Mode</p>
              <p className="text-xs text-gray-600">
                {eyeTrackingConnected ? '🟢 Connected' : eyeTrackingEnabled ? '🟡 Connecting...' : '⚪ Disabled'}
              </p>
            </div>
            <button
              onClick={() => setEyeTrackingEnabled(!eyeTrackingEnabled)}
              className={`w-14 h-8 rounded-full transition-colors ${
                eyeTrackingEnabled ? 'bg-green-500' : 'bg-gray-300'
              }`}
            >
              <motion.div
                className="w-6 h-6 bg-white rounded-full shadow-md"
                animate={{ x: eyeTrackingEnabled ? 28 : 4 }}
                transition={{ type: 'spring', stiffness: 500, damping: 30 }}
              />
            </button>
          </div>
        </div>

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
      <AnimatePresence>
        {!sessionActive && !isRecording && !showCards && !showActionButtons && !showSolution && (
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
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-6 bg-red-100 border-2 border-red-400 rounded-2xl p-4 max-w-md"
              >
                <p className="text-red-700 font-semibold text-center mb-2">⚠️ Error</p>
                <p className="text-red-600 text-sm text-center">{error}</p>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => {
                    setError("");
                    startCamera();
                  }}
                  className="mt-3 w-full bg-red-500 hover:bg-red-600 text-white font-semibold py-2 px-4 rounded-lg transition-colors"
                >
                  Try Again
                </motion.button>
              </motion.div>
            )}
          </motion.div>
        )}

        {isRecording && (
          <motion.div
            key="recording"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.5 }}
            className="flex flex-col items-center gap-6 max-w-4xl w-full"
          >
            <div className="relative bg-gray-800 rounded-3xl shadow-2xl overflow-hidden border-4 border-gray-600">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full max-w-2xl rounded-2xl transform scale-x-[-1]"
                style={{ 
                  width: '640px', 
                  height: '480px',
                  objectFit: 'cover'
                }}
              />
              
              <div className="absolute top-6 right-6 bg-black/70 text-white px-4 py-2 rounded-full text-xl font-bold shadow-lg">
                {countdown}s
              </div>

              <div className="absolute top-6 left-6 flex items-center gap-2 bg-green-500 text-white px-4 py-2 rounded-full text-sm font-semibold shadow-lg">
                <div className="w-3 h-3 bg-white rounded-full animate-pulse"></div>
                Live
              </div>
            </div>

            <canvas ref={canvasRef} style={{ display: "none" }} />

            {emotion && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                className="bg-white/90 backdrop-blur-md rounded-3xl px-16 py-12 shadow-2xl"
              >
                <div className="flex flex-col items-center gap-3">
                  <h2 
                    className="text-6xl font-bold capitalize text-center"
                    style={{
                      color: EMOTION_COLORS[emotion.toLowerCase()] || "#808080",
                      textShadow: `0 0 30px ${EMOTION_COLORS[emotion.toLowerCase()] || "#808080"}40`
                    }}
                  >
                    {emotion}
                  </h2>
                  <p className="text-2xl text-gray-600 font-semibold">
                    {(confidence * 100).toFixed(1)}%
                  </p>
                </div>
              </motion.div>
            )}

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

        {showCards && !loadingPrompts && !showActionButtons && !showSolution && (
          <motion.div
            key="cards"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -30 }}
            className="w-full flex flex-col items-center gap-6"
          >
            <PictureCards 
              prompts={prompts} 
              onSelect={handleCardSelect}
              emotion={emotion}
              gazeData={eyeTrackingEnabled ? gazeData : null}
            />
            
            {showRegenerateButton && promptType === "initial" && (
              <RegenerateButton 
                onRegenerate={handleRegenerateProblems}
                loading={loadingPrompts}
                gazeData={eyeTrackingEnabled ? gazeData : null}
              />
            )}
          </motion.div>
        )}

        {showActionButtons && !showSolution && !loadingPrompts && (
          <motion.div
            key="actions"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -30 }}
            className="w-full flex justify-center"
          >
            <ActionButtons
              onProceedToSolution={handleProceedToSolution}
              onDigDeeper={handleDigDeeper}
              gazeData={eyeTrackingEnabled ? gazeData : null}
            />
          </motion.div>
        )}

        {showSolution && !loadingPrompts && (
          <motion.div
            key="solution"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -30 }}
            className="w-full flex justify-center"
          >
            <SolutionDisplay
              solution={solution}
              emotion={emotion}
              onSatisfied={handleSatisfied}
              onNotSatisfied={handleNotSatisfied}
              gazeData={eyeTrackingEnabled ? gazeData : null}
            />
          </motion.div>
        )}

        {loadingPrompts && (
          <motion.div
            key="loading"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex flex-col items-center gap-4"
          >
            <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-purple-600"></div>
            <p className="text-xl text-gray-800">Thinking...</p>
          </motion.div>
        )}

        {error && !isRecording && !loadingPrompts && (
          <motion.div
            key="error"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-2xl w-full bg-red-100 border-2 border-red-400 rounded-3xl p-8 shadow-xl"
          >
            <div className="flex flex-col items-center gap-4">
              <div className="text-6xl">⚠️</div>
              <h2 className="text-2xl font-bold text-red-700">Something went wrong</h2>
              <p className="text-red-600 text-center">{error}</p>
              <div className="flex gap-4 mt-4">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => {
                    setError("");
                    if (showCards) {
                      // Retry fetching prompts
                      fetchInitialPrompts();
                    } else if (showActionButtons) {
                      setShowActionButtons(true);
                    } else if (showSolution) {
                      setShowSolution(true);
                    } else {
                      startCamera();
                    }
                  }}
                  className="bg-gradient-to-r from-purple-500 to-pink-500 text-white font-semibold text-lg px-8 py-3 rounded-full shadow-lg"
                >
                  Try Again
                </motion.button>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => {
                    setError("");
                    setShowSolution(false);
                    setShowActionButtons(false);
                    setShowCards(false);
                    setEmotion("");
                    setConfidence(0);
                    setConversationHistory([]);
                    setInteractionDepth(1);
                    setPrompts([]);
                    setSolution("");
                    setSessionId("");
                    setPromptType("initial");
                  }}
                  className="bg-gray-500 hover:bg-gray-600 text-white font-semibold text-lg px-8 py-3 rounded-full shadow-lg"
                >
                  Start Over
                </motion.button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Hidden video and canvas for eye tracking */}
      <video ref={eyeVideoRef} className="hidden" autoPlay playsInline muted />
      <canvas ref={eyeCanvasRef} className="hidden" />
    </div>
  );
}
