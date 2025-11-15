"use client";
import { motion } from "framer-motion";

interface RegenerateButtonProps {
  onRegenerate: () => void;
  loading: boolean;
}

export default function RegenerateButton({ onRegenerate, loading }: RegenerateButtonProps) {
  return (
    <motion.button
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      onClick={onRegenerate}
      disabled={loading}
      className="bg-gradient-to-r from-orange-400 to-amber-500 text-white font-semibold text-lg px-8 py-4 rounded-full shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
    >
      {loading ? (
        <span className="flex items-center gap-2">
          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
          Generating...
        </span>
      ) : (
        "Not listed? Show me different problems"
      )}
    </motion.button>
  );
}
