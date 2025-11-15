/**
 * Image URL mapping for picture cards
 * Uses free icon/image services
 */

/**
 * Option 1: Use Unsplash for real photos
 * Free, high-quality images
 */
export function getUnsplashImage(keyword: string): string {
  const cleanKeyword = keyword.toLowerCase().replace(/[^a-z0-9]/g, '-');
  return `https://source.unsplash.com/400x400/?${cleanKeyword},child,illustration`;
}

/**
 * Option 2: Use Twemoji (Twitter Emoji as SVG)
 * Consistent, colorful, child-friendly
 */
export function getTwemojiUrl(emoji: string): string {
  const codePoint = emoji.codePointAt(0)?.toString(16);
  return `https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/svg/${codePoint}.svg`;
}

/**
 * Option 3: Use OpenMoji (Open-source emoji)
 * Consistent style, accessible
 */
export function getOpenMojiUrl(emoji: string): string {
  const codePoint = emoji.codePointAt(0)?.toString(16).toUpperCase().padStart(4, '0');
  return `https://cdn.jsdelivr.net/npm/openmoji@latest/color/svg/${codePoint}.svg`;
}

/**
 * Option 4: Use local images
 * Store images in public/images/cards/
 */
export function getLocalImage(keyword: string): string {
  const cleanKeyword = keyword.toLowerCase().replace(/[^a-z0-9]/g, '-');
  return `/images/cards/${cleanKeyword}.png`;
}

/**
 * Recommended: Use Twemoji for consistency
 */
export function getCardImage(label: string, emoji: string): string {
  // Use Twemoji SVG for better quality
  return getTwemojiUrl(emoji);
}
