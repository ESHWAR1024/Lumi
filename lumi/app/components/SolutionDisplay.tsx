"use client";
import { motion } from "framer-motion";

interface GazeData {
  card_index: number;
  progress: number;
  gaze_x: number;
  gaze_y: number;
  face_detected: boolean;
}

interface SolutionDisplayProps {
  solution: string;
  emotion: string;
  onSatisfied: () => void;
  onNotSatisfied: () => void;
  gazeData?: GazeData | null;
}

export default function SolutionDisplay({ 
  solution, 
  emotion, 
  onSatisfied, 
  onNotSatisfied,
  gazeData
}: SolutionDisplayProps) {
  const thisHelpsProgress = gazeData?.card_index === 6 ? gazeData.progress : 0;
  const tryAgainProgress = gazeData?.card_index === 7 ? gazeData.progress : 0;
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full max-w-3xl"
    >
      <div className="bg-white/90 backdrop-blur-md rounded-3xl p-10 shadow-2xl">
        <h2 className="text-4xl font-bold text-center mb-6 text-purple-600">
          💡 Here's What We Can Do
        </h2>
        
        <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-2xl p-8 mb-8">
          <p className="text-xl text-gray-800 leading-relaxed whitespace-pre-wrap">
            {solution}
          </p>
        </div>
        
        <div className="flex gap-6 justify-center">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={onSatisfied}
            className="relative flex-1 bg-gradient-to-r from-green-400 to-green-600 text-white font-bold text-xl py-6 px-8 rounded-2xl shadow-xl overflow-hidden"
          >
            {/* Eye tracking progress indicator */}
            {thisHelpsProgress > 0 && (
              <motion.div
                className="absolute inset-0 bg-white/30"
                initial={{ width: 0 }}
                animate={{ width: `${thisHelpsProgress}%` }}
                transition={{ duration: 0.1 }}
              />
            )}
            <span className="relative z-10">😊 This Helps!</span>
          </motion.button>
          
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={onNotSatisfied}
            className="relative flex-1 bg-gradient-to-r from-orange-400 to-orange-600 text-white font-bold text-xl py-6 px-8 rounded-2xl shadow-xl overflow-hidden"
          >
            {/* Eye tracking progress indicator */}
            {tryAgainProgress > 0 && (
              <motion.div
                className="absolute inset-0 bg-white/30"
                initial={{ width: 0 }}
                animate={{ width: `${tryAgainProgress}%` }}
                transition={{ duration: 0.1 }}
              />
            )}
            <span className="relative z-10">🤔 Try Again</span>
          </motion.button>
        </div>
      </div>
    </motion.div>
  );
}
