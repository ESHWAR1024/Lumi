"use client";
import { motion } from "framer-motion";

interface GazeData {
  card_index: number;
  progress: number;
  gaze_x: number;
  gaze_y: number;
  face_detected: boolean;
}

interface RegenerateButtonProps {
  onRegenerate: () => void;
  loading: boolean;
  gazeData?: GazeData | null;
}

export default function RegenerateButton({ onRegenerate, loading, gazeData }: RegenerateButtonProps) {
  const regenerateProgress = gazeData?.card_index === 8 ? gazeData.progress : 0;
  
  return (
    <motion.button
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      onClick={onRegenerate}
      disabled={loading}
      className="relative bg-gradient-to-r from-orange-400 to-amber-500 text-white font-semibold text-lg px-8 py-4 rounded-full shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-opacity overflow-hidden"
    >
      {/* Eye tracking progress indicator */}
      {regenerateProgress > 0 && !loading && (
        <motion.div
          className="absolute inset-0 bg-white/30 rounded-full"
          initial={{ width: 0 }}
          animate={{ width: `${regenerateProgress}%` }}
          transition={{ duration: 0.1 }}
        />
      )}
      <span className="relative z-10">
        {loading ? (
          <span className="flex items-center gap-2">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
            Generating...
          </span>
        ) : (
          "Not listed? Show me different problems"
        )}
      </span>
    </motion.button>
  );
}
