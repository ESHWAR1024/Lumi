"use client";
import { motion } from "framer-motion";

interface GazeData {
  card_index: number;
  progress: number;
  gaze_x: number;
  gaze_y: number;
  face_detected: boolean;
}

interface ActionButtonsProps {
  onProceedToSolution: () => void;
  onDigDeeper: () => void;
  gazeData?: GazeData | null;
}

export default function ActionButtons({ onProceedToSolution, onDigDeeper, gazeData }: ActionButtonsProps) {
  const proceedProgress = gazeData?.card_index === 4 ? gazeData.progress : 0;
  const digDeeperProgress = gazeData?.card_index === 5 ? gazeData.progress : 0;
  
  return (
    <div className="flex gap-6 justify-center w-full max-w-2xl">
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={onProceedToSolution}
        className="relative flex-1 bg-gradient-to-r from-green-400 to-green-600 text-white font-bold text-xl py-6 px-8 rounded-2xl shadow-xl overflow-hidden"
      >
        {/* Eye tracking progress indicator */}
        {proceedProgress > 0 && (
          <motion.div
            className="absolute inset-0 bg-white/30"
            initial={{ width: 0 }}
            animate={{ width: `${proceedProgress}%` }}
            transition={{ duration: 0.1 }}
          />
        )}
        <span className="relative z-10">✅ Proceed to Solution</span>
      </motion.button>
      
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={onDigDeeper}
        className="relative flex-1 bg-gradient-to-r from-blue-400 to-blue-600 text-white font-bold text-xl py-6 px-8 rounded-2xl shadow-xl overflow-hidden"
      >
        {/* Eye tracking progress indicator */}
        {digDeeperProgress > 0 && (
          <motion.div
            className="absolute inset-0 bg-white/30"
            initial={{ width: 0 }}
            animate={{ width: `${digDeeperProgress}%` }}
            transition={{ duration: 0.1 }}
          />
        )}
        <span className="relative z-10">🔍 Dig Deeper</span>
      </motion.button>
    </div>
  );
}
