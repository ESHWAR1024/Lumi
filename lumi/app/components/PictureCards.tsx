"use client";
import { motion } from "framer-motion";

interface PromptOption {
  label: string;
  description: string;
}

interface PictureCardsProps {
  prompts: PromptOption[];
  onSelect: (selectedLabel: string) => void;
  emotion: string;
}

export default function PictureCards({ prompts, onSelect, emotion }: PictureCardsProps) {
  return (
    <div className="w-full max-w-6xl">
      <motion.h3
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-3xl font-bold text-center mb-8 text-gray-800"
      >
        Why are you feeling {emotion}?
      </motion.h3>
      
      <div className="grid grid-cols-2 gap-6">
        {prompts.map((prompt, index) => (
          <motion.button
            key={index}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.1 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => onSelect(prompt.label)}
            className="bg-white/90 backdrop-blur-md rounded-2xl p-8 shadow-xl hover:shadow-2xl transition-all"
          >
            <div className="flex flex-col items-center gap-4">
              <div className="w-32 h-32 bg-gradient-to-br from-purple-200 to-pink-200 rounded-xl flex items-center justify-center">
                <span className="text-6xl">🖼️</span>
              </div>
              <h4 className="text-2xl font-bold text-gray-800 text-center">
                {prompt.label}
              </h4>
              <p className="text-sm text-gray-600 text-center">
                {prompt.description}
              </p>
            </div>
          </motion.button>
        ))}
      </div>
    </div>
  );
}
