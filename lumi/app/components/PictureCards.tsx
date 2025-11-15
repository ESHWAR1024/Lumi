"use client";
import { motion } from "framer-motion";
import { getCardIcon } from "@/lib/cardIcons";

interface PromptOption {
  label: string;
  description: string;
}

interface GazeData {
  card_index: number;
  progress: number;
  face_detected: boolean;
}

interface PictureCardsProps {
  prompts: PromptOption[];
  onSelect: (selectedLabel: string) => void;
  emotion: string;
  gazeData?: GazeData | null;
}

export default function PictureCards({ prompts, onSelect, emotion, gazeData }: PictureCardsProps) {
  return (
    <div className="w-full max-w-3xl">
      <motion.h3
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-3xl font-bold text-center mb-8 text-gray-800"
      >
        Why are you feeling {emotion}?
      </motion.h3>
      
      <div className="flex flex-col gap-4">
        {prompts.map((prompt, index) => {
          const isGazed = gazeData?.card_index === index;
          const progress = isGazed ? gazeData.progress : 0;
          
          return (
            <motion.button
              key={index}
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => onSelect(prompt.label)}
              className={`relative bg-white/90 backdrop-blur-md rounded-2xl p-6 shadow-xl hover:shadow-2xl transition-all ${
                isGazed ? 'ring-4 ring-yellow-400 ring-offset-2 bg-yellow-50/90' : ''
              }`}
            >
              {/* Eye tracking progress bar */}
              {isGazed && (
                <div className="absolute bottom-0 left-0 right-0 h-2 bg-gray-200 rounded-b-2xl overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-green-400 to-green-600"
                    initial={{ width: 0 }}
                    animate={{ width: `${progress}%` }}
                    transition={{ duration: 0.1 }}
                  />
                </div>
              )}
              
              <div className="flex items-center gap-6">
                <div className="w-24 h-24 bg-gradient-to-br from-purple-200 to-pink-200 rounded-xl flex items-center justify-center flex-shrink-0">
                  <span className="text-6xl drop-shadow-lg">{getCardIcon(prompt.label)}</span>
                </div>
                <div className="flex-1 text-left">
                  <h4 className="text-2xl font-bold text-gray-800 mb-2">
                    {prompt.label}
                  </h4>
                  <p className="text-base text-gray-600">
                    {prompt.description}
                  </p>
                </div>
              </div>
            </motion.button>
          );
        })}
      </div>
    </div>
  );
}
