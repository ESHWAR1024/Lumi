# Eye Tracking Button Support

## Overview
Extended the eye tracking system to support selecting action buttons (Proceed to Solution and Dig Deeper) and solution feedback buttons (This Helps and Try Again) in addition to picture cards.

## Changes Made

### Backend (eye_tracking_service.py)

1. **New Function: `map_gaze_to_button()`**
   - Maps gaze coordinates to action buttons
   - Button 4 = Proceed to Solution (left)
   - Button 5 = Dig Deeper (right)
   - Uses Y threshold of 0.55 to detect button zone
   - Horizontal hysteresis of 0.10 for stable selection

2. **New Function: `map_gaze_to_solution_button()`**
   - Maps gaze coordinates to solution feedback buttons
   - Button 6 = This Helps (left)
   - Button 7 = Try Again (right)
   - Uses same Y threshold and hysteresis as action buttons

3. **Updated WebSocket Handler**
   - Now accepts `mode` parameter in frame messages ("cards", "buttons", or "solution")
   - Routes to appropriate mapping function based on mode
   - Maintains same dwell time (5 seconds) for all modes

### Frontend

1. **useSharedEyeTracking Hook (useSharedEyeTracking.ts)** - NEW
   - Shared WebSocket connection and camera stream across all instances
   - Added `mode` parameter: 'cards' | 'buttons' | 'solution'
   - Dynamically switches mode based on what's visible
   - Prevents conflicts from multiple connections
   - Single camera stream shared by all components

2. **ActionButtons Component (ActionButtons.tsx)**
   - Added `gazeData` prop for eye tracking integration
   - Shows progress indicator (white overlay) when button is being gazed at
   - Button 4 (Proceed) shows progress on left button
   - Button 5 (Dig Deeper) shows progress on right button

3. **SolutionDisplay Component (SolutionDisplay.tsx)**
   - Added `gazeData` prop for eye tracking integration
   - Shows progress indicator (white overlay) when button is being gazed at
   - Button 6 (This Helps) shows progress on left button
   - Button 7 (Try Again) shows progress on right button

4. **Start Page (start/page.tsx)**
   - Uses single shared eye tracking instance
   - Dynamically switches mode based on visible components:
     - `showSolution` → 'solution' mode
     - `showActionButtons` → 'buttons' mode
     - `showCards` → 'cards' mode
   - Unified `handleEyeTrackingSelect()` handler for:
     - Cards (0-3)
     - Action buttons (4-5)
     - Solution buttons (6-7)
   - Passes same gaze data to PictureCards, ActionButtons, and SolutionDisplay components

## How It Works

1. Single shared WebSocket connection and camera stream for all eye tracking
2. Mode automatically switches based on what's visible:
   - Picture cards → 'cards' mode (indices 0-3)
   - Action buttons → 'buttons' mode (indices 4-5)
   - Solution feedback → 'solution' mode (indices 6-7)
3. User looks at any button for 5 seconds
4. Backend detects gaze in button zone (Y > 0.55) and determines left/right
5. Progress bar fills up as white overlay on the button
6. When complete, the corresponding action is triggered automatically

**Key Fix:** Using a shared connection prevents conflicts from multiple WebSocket instances trying to access the camera simultaneously.

## Testing

1. Start the eye tracking service:
   ```bash
   cd lumi/lumi_backend/eye_tracking
   python eye_tracking_service.py
   ```

2. Enable eye tracking in the app menu
3. Complete emotion detection and select a problem card
4. When action buttons appear, look at either button for 5 seconds to select

## Button Layout

### Action Buttons (after selecting a card)
```
Screen Layout:
┌─────────────────────────┐
│                         │
│   [Picture Cards]       │  Y < 0.55
│                         │
├─────────────────────────┤
│  [Proceed] [Dig Deeper] │  Y > 0.55
│     (4)        (5)      │
└─────────────────────────┘
     X < 0.5   X > 0.5
```

### Solution Buttons (after solution is shown)
```
Screen Layout:
┌─────────────────────────┐
│                         │
│   [Solution Text]       │  Y < 0.55
│                         │
├─────────────────────────┤
│ [This Helps][Try Again] │  Y > 0.55
│     (6)        (7)      │
└─────────────────────────┘
     X < 0.5   X > 0.5
```

## Accuracy Settings

Same optimized settings as card selection:
- Sensitivity X: 22.0
- Sensitivity Y: 20.0
- Smoothing window: 7 frames
- Hysteresis: 0.08 (vertical), 0.10 (horizontal for buttons)
- Dwell time: 5.0 seconds (unchanged)
