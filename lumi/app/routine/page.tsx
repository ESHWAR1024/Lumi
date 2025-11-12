"use client";
import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { useRouter } from "next/navigation";
import { supabase, ChildRoutine } from "@/lib/supabase";

export default function RoutinePage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [formData, setFormData] = useState({
    wake_time: "",
    breakfast_time: "",
    lunch_time: "",
    dinner_time: "",
    snack_time_1: "",
    snack_time_2: "",
    nap_time: "",
    bedtime: "",
    favorite_activities: "",
    comfort_items: "",
    triggers_to_avoid: "",
    preferred_prompts: "",
  });

  useEffect(() => {
    const profileId = localStorage.getItem("childProfileId");
    if (!profileId) {
      router.push("/onboarding");
    }
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const profileId = localStorage.getItem("childProfileId");
      if (!profileId) throw new Error("No profile found");

      const snackTimes = [formData.snack_time_1, formData.snack_time_2].filter(
        (time) => time !== ""
      );

      const routineData: ChildRoutine = {
        child_profile_id: profileId,
        wake_time: formData.wake_time,
        breakfast_time: formData.breakfast_time,
        lunch_time: formData.lunch_time,
        dinner_time: formData.dinner_time,
        snack_times: snackTimes,
        nap_time: formData.nap_time,
        bedtime: formData.bedtime,
        favorite_activities: formData.favorite_activities
          .split(",")
          .map((item) => item.trim())
          .filter((item) => item !== ""),
        comfort_items: formData.comfort_items
          .split(",")
          .map((item) => item.trim())
          .filter((item) => item !== ""),
        triggers_to_avoid: formData.triggers_to_avoid
          .split(",")
          .map((item) => item.trim())
          .filter((item) => item !== ""),
        preferred_prompts: formData.preferred_prompts
          .split(",")
          .map((item) => item.trim())
          .filter((item) => item !== ""),
      };

      const { error } = await supabase
        .from("child_routines")
        .insert([routineData]);

      if (error) throw error;

      localStorage.setItem("routineCompleted", "true");
      router.push("/start");
    } catch (err: any) {
      setError(err.message || "Failed to save routine");
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
        className="bg-white/80 backdrop-blur-md rounded-3xl shadow-2xl p-8 w-full max-w-3xl max-h-[90vh] overflow-y-auto"
      >
        <h1 className="text-4xl font-bold text-center mb-2 text-[#2E2E2E]">
          Daily Routine 📅
        </h1>
        <p className="text-center text-gray-600 mb-6">
          Help Lumi understand your child's day
        </p>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Daily Schedule */}
          <div>
            <h2 className="text-xl font-semibold mb-3 text-[#2E2E2E]">
              ⏰ Daily Schedule
            </h2>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Wake Up Time
                </label>
                <input
                  type="time"
                  required
                  value={formData.wake_time}
                  onChange={(e) =>
                    setFormData({ ...formData, wake_time: e.target.value })
                  }
                  className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#A2D2FF] outline-none text-black"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Breakfast Time
                </label>
                <input
                  type="time"
                  required
                  value={formData.breakfast_time}
                  onChange={(e) =>
                    setFormData({ ...formData, breakfast_time: e.target.value })
                  }
                  className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#A2D2FF] outline-none text-black"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Lunch Time
                </label>
                <input
                  type="time"
                  required
                  value={formData.lunch_time}
                  onChange={(e) =>
                    setFormData({ ...formData, lunch_time: e.target.value })
                  }
                  className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#A2D2FF] outline-none text-black"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Dinner Time
                </label>
                <input
                  type="time"
                  required
                  value={formData.dinner_time}
                  onChange={(e) =>
                    setFormData({ ...formData, dinner_time: e.target.value })
                  }
                  className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#A2D2FF] outline-none text-black"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Snack Time 1 (Optional)
                </label>
                <input
                  type="time"
                  value={formData.snack_time_1}
                  onChange={(e) =>
                    setFormData({ ...formData, snack_time_1: e.target.value })
                  }
                  className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#A2D2FF] outline-none text-black"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Snack Time 2 (Optional)
                </label>
                <input
                  type="time"
                  value={formData.snack_time_2}
                  onChange={(e) =>
                    setFormData({ ...formData, snack_time_2: e.target.value })
                  }
                  className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#A2D2FF] outline-none text-black"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Nap Time
                </label>
                <input
                  type="time"
                  required
                  value={formData.nap_time}
                  onChange={(e) =>
                    setFormData({ ...formData, nap_time: e.target.value })
                  }
                  className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#A2D2FF] outline-none text-black"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Bedtime
                </label>
                <input
                  type="time"
                  required
                  value={formData.bedtime}
                  onChange={(e) =>
                    setFormData({ ...formData, bedtime: e.target.value })
                  }
                  className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#A2D2FF] outline-none text-black"
                />
              </div>
            </div>
          </div>

          {/* Favorite Activities */}
          <div>
            <h2 className="text-xl font-semibold mb-3 text-[#2E2E2E]">
              🎨 Favorite Activities
            </h2>
            <input
              type="text"
              required
              value={formData.favorite_activities}
              onChange={(e) =>
                setFormData({ ...formData, favorite_activities: e.target.value })
              }
              className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#A2D2FF] outline-none text-black"
              placeholder="e.g., playing with toys, listening to music, outdoor time (comma separated)"
            />
          </div>

          {/* Comfort Items */}
          <div>
            <h2 className="text-xl font-semibold mb-3 text-[#2E2E2E]">
              🧸 Comfort Items (Optional)
            </h2>
            <input
              type="text"
              value={formData.comfort_items}
              onChange={(e) =>
                setFormData({ ...formData, comfort_items: e.target.value })
              }
              className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#A2D2FF] outline-none text-black"
              placeholder="e.g., teddy bear, blanket, music box (comma separated)"
            />
          </div>

          {/* Triggers to Avoid */}
          <div>
            <h2 className="text-xl font-semibold mb-3 text-[#2E2E2E]">
              ⚠️ Triggers to Avoid (Optional)
            </h2>
            <input
              type="text"
              value={formData.triggers_to_avoid}
              onChange={(e) =>
                setFormData({ ...formData, triggers_to_avoid: e.target.value })
              }
              className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#A2D2FF] outline-none text-black"
              placeholder="e.g., loud noises, bright lights, crowds (comma separated)"
            />
          </div>

          {/* Preferred Prompt Categories */}
          <div>
            <h2 className="text-xl font-semibold mb-3 text-[#2E2E2E]">
              🖼️ Preferred Picture Prompts (Optional)
            </h2>
            <input
              type="text"
              value={formData.preferred_prompts}
              onChange={(e) =>
                setFormData({ ...formData, preferred_prompts: e.target.value })
              }
              className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#A2D2FF] outline-none text-black"
              placeholder="e.g., food, toys, emotions, activities (comma separated)"
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
            {loading ? "Saving..." : "Complete Setup →"}
          </motion.button>
        </form>
      </motion.div>
    </div>
  );
}
