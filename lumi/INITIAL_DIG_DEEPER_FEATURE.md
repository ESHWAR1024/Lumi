# Initial "Dig Deeper" Feature

## What Changed

Added a "None of these - Show me different options" button after the initial 4 emotion cards, allowing children to get alternative suggestions if the first set doesn't match their situation.

---

## How It Works

### Before (Old Flow):
```
1. Camera detects emotion (e.g., "sad")
2. Shows 4 initial cards guessing why
3. Child MUST select one card
4. Shows follow-up prompts
5. Action buttons (Dig Deeper / Proceed to Solution)
```

### After (New Flow):
```
1. Camera detects emotion (e.g., "sad")
2. Shows 4 initial cards guessing why
3. Child can either:
   a) Select a card → Continue to follow-up prompts
   b) Click "None of these" → Get 4 NEW alternative cards
4. If clicked "None of these":
   - Generates 4 completely different initial prompts
   - Child can select from new cards OR click "None of these" again
5. Once card selected → Shows follow-up prompts
6. Action buttons (Dig Deeper / Proceed to Solution)
```

---

## Visual Flow

```
┌─────────────────────────────────┐
│   Emotion Detected: Sad 😢      │
└─────────────────────────────────┘
              ↓
┌─────────────────────────────────┐
│  Initial 4 Cards:               │
│  • Hungry?                      │
│  • Tired?                       │
│  • Missing someone?             │
│  • Uncomfortable?               │
└─────────────────────────────────┘
              ↓
        ┌─────┴─────┐
        │           │
    Select      Click "None
     Card       of these"
        │           │
        │           ↓
        │    ┌─────────────────┐
        │    │ NEW 4 Cards:    │
        │    │ • Bored?        │
        │    │ • Frustrated?   │
        │    │ • Scared?       │
        │    │ • Lonely?       │
        │    └─────────────────┘
        │           │
        └─────┬─────┘
              ↓
    ┌─────────────────────┐
    │  Follow-up Prompts  │
    └─────────────────────┘
              ↓
    ┌─────────────────────┐
    │  Action Buttons     │
    │  • Dig Deeper       │
    │  • Proceed          │
    └─────────────────────┘
```

---

## Technical Implementation

### Frontend Changes (`lumi/app/start/page.tsx`)

1. **New Function: `handleInitialDigDeeper()`**
   - Calls the same `/api/prompts/initial` endpoint
   - Generates fresh alternative prompts
   - Stores interaction as "initial_dig_deeper"

2. **UI Addition**
   - Button appears ONLY when `promptType === "initial"`
   - Styled with purple-to-pink gradient
   - Text: "None of these - Show me different options"

3. **Flow Logic**
   - Button is hidden after first card selection
   - Can be clicked multiple times for more alternatives
   - Doesn't affect the rest of the conversation flow

### Backend (No Changes Needed!)

The existing `/api/prompts/initial` endpoint already:
- Generates context-aware prompts
- Uses Gemini AI for variety
- Each call produces different suggestions naturally

---

## User Experience

### Scenario 1: First Set Works
```
Child sees: "Hungry?" → Clicks it → Continues normally
```

### Scenario 2: First Set Doesn't Match
```
Child sees: "Hungry?", "Tired?", "Missing someone?", "Uncomfortable?"
↓
None match → Clicks "None of these"
↓
New cards: "Bored?", "Frustrated?", "Scared?", "Lonely?"
↓
Clicks "Frustrated?" → Continues normally
```

### Scenario 3: Multiple Attempts
```
First set → "None of these"
Second set → "None of these"
Third set → Finds match → Continues
```

---

## Benefits

1. **Better Accuracy** - More chances to identify the real issue
2. **Child Empowerment** - Child has control to find the right match
3. **Reduced Frustration** - No forced selection of wrong option
4. **Natural Flow** - Doesn't disrupt existing conversation logic
5. **Unlimited Attempts** - Can keep trying until they find a match

---

## Database Tracking

Interactions are stored with:
- `action_type: "initial_dig_deeper"`
- `selected_option: "none_of_these"`
- `prompt_options: [new set of 4 cards]`

This helps track:
- How often children need alternatives
- Which emotions need better initial prompts
- Patterns in prompt effectiveness

---

## Testing

### Test Case 1: Happy Path
1. Start camera
2. Emotion detected
3. See 4 initial cards
4. Click "None of these"
5. Verify: New 4 cards appear
6. Select one card
7. Verify: Follow-up prompts appear
8. Continue normally

### Test Case 2: Multiple Alternatives
1. Get initial cards
2. Click "None of these" → Get set 2
3. Click "None of these" → Get set 3
4. Select a card from set 3
5. Verify: Flow continues normally

### Test Case 3: Direct Selection
1. Get initial cards
2. Select a card immediately (don't click "None of these")
3. Verify: Button disappears
4. Verify: Follow-up prompts appear
5. Verify: Rest of flow unchanged

---

## Important Notes

- ✅ **No backend changes needed** - Uses existing API
- ✅ **No database schema changes** - Uses existing tables
- ✅ **Doesn't affect other steps** - Only adds option to initial prompts
- ✅ **Can be clicked multiple times** - Unlimited alternatives
- ✅ **Gemini naturally varies responses** - Each call gives different prompts

---

## Future Enhancements (Optional)

1. **Limit attempts** - Max 3 "None of these" clicks
2. **Learn from patterns** - Track which alternatives work best
3. **Smarter alternatives** - Use previous rejected options to inform new ones
4. **Visual feedback** - Show "Attempt 2 of 3" counter
5. **Fallback option** - After 3 attempts, offer "Type your own reason"

---

**This feature is now live! Try it out by starting a session and clicking "None of these" after the initial cards appear.** 🎉
