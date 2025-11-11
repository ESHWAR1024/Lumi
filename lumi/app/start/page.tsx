"use client";
import { useState } from "react";
import { motion } from "framer-motion";

export default function StartPage() {
  const [menuOpen, setMenuOpen] = useState(false);

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
        initial={{ x: -200, opacity: 0 }}
        animate={menuOpen ? { x: 0, opacity: 1 } : { x: -200, opacity: 0 }}
        transition={{ duration: 0.5 }}
        className="absolute top-0 left-0 h-full w-52 bg-black/70 text-white p-6 pt-20 flex flex-col space-y-6 shadow-lg"
      >
        <p className="text-lg">Weekly Report 1</p>
        <p className="text-lg">Weekly Report 2</p>
        <p className="text-lg">Weekly Report 3</p>
        <p className="text-lg">Weekly Report 4</p>
      </motion.div>

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
