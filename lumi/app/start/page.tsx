"use client";
import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { supabase, ChildProfile } from "@/lib/supabase";
import { useRouter } from "next/navigation";

export default function StartPage() {
  const router = useRouter();
  const [menuOpen, setMenuOpen] = useState(false);
  const [childProfile, setChildProfile] = useState<ChildProfile | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadProfile();
  }, []);

  const loadProfile = async () => {
    const storedProfileId = localStorage.getItem("childProfileId");
    if (!storedProfileId) {
      router.push("/onboarding");
      return;
    }

    // Check if routine is completed
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
        {/* Child Profile Section */}
        {childProfile && (
          <div className="mb-8 pb-6 border-b-2 border-gray-200">
            <h2 className="text-xl font-bold">{childProfile.child_name}</h2>
            <p className="text-sm text-gray-600">{childProfile.age} years old</p>
          </div>
        )}

        {/* Weekly Reports Section */}
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

      {/* Overlay to close menu when clicking outside */}
      {menuOpen && (
        <div
          className="absolute inset-0 bg-black/20 z-0"
          onClick={() => setMenuOpen(false)}
        />
      )}

      {/* Main Content */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1 }}
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
          className="bg-gradient-to-r from-[#A2D2FF] to-[#FFC8DD] text-[#2E2E2E] font-semibold text-2xl w-32 h-32 rounded-full shadow-xl flex items-center justify-center"
        >
          Start
        </motion.button>
      </motion.div>
    </div>
  );
}
