"use client";
import { motion } from "framer-motion";
import { useRouter } from "next/navigation";

export default function HomePage() {
  const router = useRouter();

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-[#A2D2FF] via-[#FFC8DD] to-[#FFAFCC] flex flex-col items-center justify-center text-[#2E2E2E]">
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1 }}
        className="flex flex-col items-center justify-center w-full px-8 py-16"
      >
        <h1 className="text-7xl font-extrabold mb-4 tracking-wide">Lumi 🌟</h1>
        <p className="text-lg mb-8 text-center max-w-lg">
          Your empathetic AI companion — connecting children, caregivers, and therapy through emotion, understanding, and light.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 w-full max-w-5xl">
          <motion.div
            whileHover={{ scale: 1.05 }}
            className="bg-white/70 backdrop-blur-md rounded-2xl shadow-lg p-8 text-center"
          >
            <h2 className="text-2xl font-bold mb-2">👶 Child Profile</h2>
            <p className="text-gray-700">
              Create or view your child’s profile and customize Lumi’s interaction style for comfort.
            </p>
          </motion.div>

          <motion.div
            whileHover={{ scale: 1.05 }}
            className="bg-white/70 backdrop-blur-md rounded-2xl shadow-lg p-8 text-center"
          >
            <h2 className="text-2xl font-bold mb-2">💬 Companion Mode</h2>
            <p className="text-gray-700">
              Activate Lumi’s live emotion detection and visual communication system to assist your child.
            </p>
          </motion.div>

          <motion.div
            whileHover={{ scale: 1.05 }}
            className="bg-white/70 backdrop-blur-md rounded-2xl shadow-lg p-8 text-center"
          >
            <h2 className="text-2xl font-bold mb-2">📊 Parent Dashboard</h2>
            <p className="text-gray-700">
              Track emotional trends, communication patterns, and therapy insights all in one place.
            </p>
          </motion.div>
        </div>

        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => router.push("/start")} // Navigate to /start
          className="mt-10 bg-gradient-to-r from-[#A2D2FF] to-[#FFC8DD] text-[#2E2E2E] font-semibold py-3 px-8 rounded-full shadow-lg"
        >
          Get Started →
        </motion.button>
      </motion.div>
    </div>
  );
}
