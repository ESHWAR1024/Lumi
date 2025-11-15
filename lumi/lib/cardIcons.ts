/**
 * Icon mapping for picture cards
 * Maps common card labels to relevant emojis
 */

export const CARD_ICONS: { [key: string]: string } = {
  // Food related
  "hungry": "🍽️",
  "thirsty": "💧",
  "breakfast": "🥞",
  "lunch": "🍱",
  "dinner": "🍝",
  "snack": "🍪",
  "food": "🍎",
  
  // Physical sensations
  "tired": "😴",
  "sleepy": "🛌",
  "pain": "🤕",
  "hurt": "💢",
  "uncomfortable": "😣",
  "hot": "🥵",
  "cold": "🥶",
  
  // Emotional
  "lonely": "😢",
  "scared": "😨",
  "worried": "😰",
  "frustrated": "😤",
  "bored": "😑",
  "excited": "🤩",
  "happy": "😊",
  
  // Sensory (Autism-specific)
  "loud": "🔊",
  "bright": "💡",
  "noisy": "📢",
  "quiet": "🤫",
  "dark": "🌙",
  "texture": "✋",
  
  // Activities
  "play": "🎮",
  "music": "🎵",
  "outside": "🌳",
  "toy": "🧸",
  "book": "📚",
  "tv": "📺",
  
  // People
  "mom": "👩",
  "dad": "👨",
  "friend": "👫",
  "teacher": "👩‍🏫",
  "family": "👨‍👩‍👧",
  
  // Routine
  "school": "🏫",
  "home": "🏠",
  "therapy": "🏥",
  "bedtime": "🌙",
  "bath": "🛁",
  
  // Comfort items
  "blanket": "🛏️",
  "pillow": "🛋️",
  "hug": "🤗",
  
  // Medical/Physical
  "medicine": "💊",
  "doctor": "👨‍⚕️",
  "wheelchair": "♿",
  
  // Default
  "default": "🖼️"
};

/**
 * Get icon for a card label
 * Uses fuzzy matching to find relevant icon
 */
export function getCardIcon(label: string): string {
  const lowerLabel = label.toLowerCase();
  
  // Direct match
  for (const [key, icon] of Object.entries(CARD_ICONS)) {
    if (lowerLabel.includes(key)) {
      return icon;
    }
  }
  
  // Return default if no match
  return CARD_ICONS.default;
}
