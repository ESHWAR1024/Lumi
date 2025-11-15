"use client";
import { motion } from "framer-motion";

interface ActionButtonsProps {
  onProceedToSolution: () => void;
  onDigDeeper: () => void;
}

export default function ActionButtons({ onProceedToSolution, onDigDeeper }: ActionButtonsProps) {
  return (
    <div className="flex gap-6 justify-center w-full max-w-2xl">
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={onProceedToSolution}
        className="flex-1 bg-gradient-to-r from-green-400 to-green-600 text-white font-bold text-xl py-6 px-8 rounded-2xl shadow-xl"
      >
        ✅ Proceed to Solution
      </motion.button>
      
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={onDigDeeper}
        className="flex-1 bg-gradient-to-r from-blue-400 to-blue-600 text-white font-bold text-xl py-6 px-8 rounded-2xl shadow-xl"
      >
        🔍 Dig Deeper
      </motion.button>
    </div>
  );
}
