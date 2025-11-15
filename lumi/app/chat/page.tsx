"use client";
import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { supabase, ChildProfile } from "@/lib/supabase";
import { useRouter } from "next/navigation";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

export default function ChatPage() {
  const router = useRouter();
  const [childProfile, setChildProfile] = useState<ChildProfile | null>(null);
  const [childRoutine, setChildRoutine] = useState<any>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadProfile();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const loadProfile = async () => {
    try {
      const storedProfileId = localStorage.getItem("childProfileId");
      if (!storedProfileId) {
        router.push("/onboarding");
        return;
      }

      const { data: profileData, error: profileError } = await supabase
        .from("child_profiles")
        .select("*")
        .eq("id", storedProfileId)
        .single();

      if (profileError || !profileData) {
        router.push("/onboarding");
        return;
      }

      // Check if child can type - if false, redirect to camera-based interface
      if (profileData.can_type === false) {
        router.push("/start");
        return;
      }

      setChildProfile(profileData);

      const { data: routineData } = await supabase
        .from("child_routines")
        .select("*")
        .eq("child_profile_id", storedProfileId)
        .maybeSingle();

      setChildRoutine(routineData);

      // Add welcome message
      setMessages([
        {
          id: "welcome",
          role: "assistant",
          content: `Hi ${profileData.child_name}! 👋 I'm Lumi, your AI companion. I'm here to chat with you about anything - how you're feeling, what's on your mind, or just to have a friendly conversation. What would you like to talk about today?`,
          timestamp: new Date(),
        },
      ]);
    } catch (err) {
      console.error("Error loading profile:", err);
    } finally {
      setInitialLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || loading || !childProfile) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      // Format routine for API (convert arrays to strings)
      const formatRoutineForAPI = (routine: any) => {
        if (!routine) return null;
        
        return {
          wake_up_time: routine.wake_time || routine.wake_up_time || null,
          breakfast_time: routine.breakfast_time || null,
          lunch_time: routine.lunch_time || null,
          snacks_time: routine.snack_times
            ? Array.isArray(routine.snack_times)
              ? routine.snack_times.join(", ")
              : routine.snack_times
            : null,
          dinner_time: routine.dinner_time || null,
          bedtime: routine.bedtime || null,
          favorite_activities: routine.favorite_activities
            ? Array.isArray(routine.favorite_activities)
              ? routine.favorite_activities.join(", ")
              : routine.favorite_activities
            : null,
          comfort_items: routine.comfort_items
            ? Array.isArray(routine.comfort_items)
              ? routine.comfort_items.join(", ")
              : routine.comfort_items
            : null,
          preferred_prompts: routine.preferred_prompts
            ? Array.isArray(routine.preferred_prompts)
              ? routine.preferred_prompts.join(", ")
              : routine.preferred_prompts
            : null,
          communication_preferences: routine.communication_preferences || null,
        };
      };

      const response = await fetch("http://localhost:8001/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: input.trim(),
          conversation_history: messages.map((m) => ({
            role: m.role,
            content: m.content,
          })),
          child_profile: {
            child_name: childProfile.child_name,
            age: childProfile.age,
            diagnosis: childProfile.diagnosis || childProfile.condition,
          },
          child_routine: formatRoutineForAPI(childRoutine),
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get response");
      }

      const data = await response.json();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.response,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      console.error("Error sending message:", err);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content:
          "I'm sorry, I'm having trouble responding right now. Please try again.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  if (initialLoading) {
    return (
      <div className="min-h-screen w-full bg-gradient-to-br from-[#FFC8DD] via-[#E0BBE4] to-[#A2D2FF] flex items-center justify-center">
        <p className="text-2xl text-[#2E2E2E]">Loading...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-[#FFC8DD] via-[#E0BBE4] to-[#A2D2FF] flex flex-col items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="w-full max-w-4xl h-[85vh] bg-white/80 backdrop-blur-md rounded-3xl shadow-2xl flex flex-col overflow-hidden"
      >
        {/* Header */}
        <div className="bg-gradient-to-r from-[#A2D2FF] to-[#FFC8DD] p-6 text-center">
          <h1 className="text-3xl font-bold text-[#2E2E2E]">
            Chat with Lumi 🌟
          </h1>
          {childProfile && (
            <p className="text-sm text-[#2E2E2E] mt-1">
              Hi {childProfile.child_name}! I'm here to listen.
            </p>
          )}
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          <AnimatePresence>
            {messages.map((message) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className={`flex ${
                  message.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={`max-w-[75%] rounded-2xl px-6 py-4 ${
                    message.role === "user"
                      ? "bg-gradient-to-r from-[#A2D2FF] to-[#FFC8DD] text-[#2E2E2E]"
                      : "bg-white/90 text-[#2E2E2E] shadow-md"
                  }`}
                >
                  <p className="text-lg whitespace-pre-wrap">{message.content}</p>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>

          {loading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex justify-start"
            >
              <div className="bg-white/90 rounded-2xl px-6 py-4 shadow-md">
                <div className="flex space-x-2">
                  <div className="w-3 h-3 bg-[#A2D2FF] rounded-full animate-bounce"></div>
                  <div
                    className="w-3 h-3 bg-[#FFC8DD] rounded-full animate-bounce"
                    style={{ animationDelay: "0.1s" }}
                  ></div>
                  <div
                    className="w-3 h-3 bg-[#E0BBE4] rounded-full animate-bounce"
                    style={{ animationDelay: "0.2s" }}
                  ></div>
                </div>
              </div>
            </motion.div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="p-6 bg-white/50 backdrop-blur-sm border-t border-gray-200">
          <div className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message here..."
              disabled={loading}
              className="flex-1 px-6 py-4 rounded-full border-2 border-[#A2D2FF] focus:border-[#FFC8DD] focus:outline-none text-[#2E2E2E] text-lg bg-white/90 disabled:opacity-50"
            />
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={sendMessage}
              disabled={loading || !input.trim()}
              className="bg-gradient-to-r from-[#A2D2FF] to-[#FFC8DD] text-[#2E2E2E] font-semibold px-8 py-4 rounded-full shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Send
            </motion.button>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
