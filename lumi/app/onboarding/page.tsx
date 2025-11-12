"use client";
import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { useRouter } from "next/navigation";
import { supabase, ChildProfile } from "@/lib/supabase";

export default function OnboardingPage() {
  const router = useRouter();
  const [formData, setFormData] = useState({
    child_name: "",
    age: "",
    parent_email: "",
    condition: "",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    checkExistingProfile();
  }, []);

  const checkExistingProfile = async () => {
    const storedProfileId = localStorage.getItem("childProfileId");
    if (storedProfileId) {
      const { data, error } = await supabase
        .from("child_profiles")
        .select("*")
        .eq("id", storedProfileId)
        .single();

      if (data && !error) {
        router.push("/start");
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const profileData: ChildProfile = {
        child_name: formData.child_name,
        age: parseInt(formData.age),
        parent_email: formData.parent_email,
        condition: formData.condition,
      };

      const { data, error } = await supabase
        .from("child_profiles")
        .insert([profileData])
        .select()
        .single();

      if (error) throw error;

      if (data) {
        localStorage.setItem("childProfileId", data.id);
        router.push("/routine");
      }
    } catch (err: any) {
      setError(err.message || "Failed to save profile");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-[#A2D2FF] via-[#FFC8DD] to-[#FFAFCC] flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        className="bg-white/80 backdrop-blur-md rounded-3xl shadow-2xl p-8 w-full max-w-md"
      >
        <h1 className="text-4xl font-bold text-center mb-2 text-[#2E2E2E]">
          Welcome to Lumi 🌟
        </h1>
        <p className="text-center text-gray-600 mb-6">
          Let's get to know your child
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Child's Name
            </label>
            <input
              type="text"
              required
              value={formData.child_name}
              onChange={(e) =>
                setFormData({ ...formData, child_name: e.target.value })
              }
              className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#A2D2FF] focus:border-transparent outline-none text-black"
              placeholder="Enter child's name"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Age
            </label>
            <input
              type="number"
              required
              min="1"
              max="18"
              value={formData.age}
              onChange={(e) =>
                setFormData({ ...formData, age: e.target.value })
              }
              className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#A2D2FF] focus:border-transparent outline-none text-black"
              placeholder="Enter age"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Parent's Email
            </label>
            <input
              type="email"
              required
              value={formData.parent_email}
              onChange={(e) =>
                setFormData({ ...formData, parent_email: e.target.value })
              }
              className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#A2D2FF] focus:border-transparent outline-none text-black"
              placeholder="parent@example.com"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Condition/Diagnosis
            </label>
            <input
              type="text"
              required
              value={formData.condition}
              onChange={(e) =>
                setFormData({ ...formData, condition: e.target.value })
              }
              className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#A2D2FF] focus:border-transparent outline-none text-black"
              placeholder="e.g., Nonverbal Autism, Rett Syndrome"
            />
          </div>

          {error && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-2 rounded-lg text-sm">
              {error}
            </div>
          )}

          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            type="submit"
            disabled={loading}
            className="w-full bg-gradient-to-r from-[#A2D2FF] to-[#FFC8DD] text-[#2E2E2E] font-semibold py-3 rounded-full shadow-lg disabled:opacity-50"
          >
            {loading ? "Saving..." : "Continue →"}
          </motion.button>
        </form>
      </motion.div>
    </div>
  );
}
