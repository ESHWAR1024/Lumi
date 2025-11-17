# Complete Eye Tracking Solution

## Summary
Full eye tracking implementation for Lumi that supports:
- ✅ Picture card selection (4 cards)
- ✅ Action button selection (Proceed to Solution, Dig Deeper)
- ✅ Solution feedback buttons (This Helps, Try Again)

## Index Mapping

| Index | Component | Description |
|-------|-----------|-------------|
| 0-3   | Picture Cards | Four problem identification cards |
| 4     | Action Button | Proceed to Solution (left) |
| 5     | Action Button | Dig Deeper (right) |
| 6     | Solution Button | This Helps (left) |
| 7     | Solution Button | Try Again (right) |
| 8     | Regenerate Button | Show me different problems (centered) |

## Architecture

### Single Shared Connection
- One WebSocket connection to backend
- One camera stream shared across all components
- Dynamic mode switching based on visible UI

### Mode System
```typescript
type EyeTrackingMode = 'cards' | 'buttons' | 'solution' | 'cards_with_regenerate';
```

Mode automatically switches based on:
- `showCards = true` + `showRegenerateButton = true` → 'cards_with_regenerate' mode
- `showCards = true` → 'cards' mode
- `showActionButtons = true` → 'buttons' mode
- `showSolution = true` → 'solution' mode

## Files Modified

### Backend
- `lumi/lumi_backend/eye_tracking/eye_tracking_service.py`
  - Added `map_gaze_to_button()` for action buttons
  - Added `map_gaze_to_solution_button()` for solution buttons
  - Updated WebSocket handler to support 3 modes

### Frontend Hooks
- `lumi/app/hooks/useSharedEyeTracking.ts` (NEW)
  - Shared WebSocket and camera management
  - Prevents multiple connection conflicts
  - Dynamic mode switching

### Frontend Components
- `lumi/app/components/ActionButtons.tsx`
  - Added gaze data prop
  - Progress indicators for buttons 4 & 5

- `lumi/app/components/SolutionDisplay.tsx`
  - Added gaze data prop
  - Progress indicators for buttons 6 & 7

- `lumi/app/start/page.tsx`
  - Unified eye tracking handler
  - Mode detection logic
  - Gaze data distribution to all components

## User Flow

1. **Start** → Enable eye tracking in menu
2. **Emotion Detection** → Camera captures emotion
3. **Picture Cards** (mode: 'cards_with_regenerate')
   - Look at card (0-3) for 5 seconds to select
   - OR look at "Show me different problems" button (8) for 5 seconds to regenerate
   - Progress bar fills
   - Selection/regeneration happens automatically
4. **Action Buttons** (mode: 'buttons')
   - Look at "Proceed" (4) or "Dig Deeper" (5) for 5 seconds
   - Progress bar fills
   - Action triggered automatically
5. **Solution Display** (mode: 'solution')
   - Look at "This Helps" (6) or "Try Again" (7) for 5 seconds
   - Progress bar fills
   - Feedback submitted automatically

## Accuracy Settings

All modes use the same optimized settings:
- **Sensitivity X**: 22.0 (horizontal)
- **Sensitivity Y**: 20.0 (vertical)
- **Smoothing Window**: 7 frames
- **Hysteresis**: 0.08 (vertical), 0.10 (horizontal)
- **Dwell Time**: 5.0 seconds
- **Transition Time**: 0.2 seconds

## Testing

1. Start eye tracking service:
```bash
cd lumi/lumi_backend/eye_tracking
python eye_tracking_service.py
```

2. Start frontend:
```bash
cd lumi
npm run dev
```

3. Test flow:
   - Enable eye tracking in menu
   - Complete emotion detection
   - Select a problem card with eyes
   - Select an action button with eyes
   - Select a solution feedback button with eyes

## Troubleshooting

### Eye tracking not working
- Check WebSocket connection (port 8002)
- Verify camera permissions
- Check browser console for errors

### Mode not switching
- Verify `showCards`, `showActionButtons`, `showSolution` states
- Check console logs for mode switch messages

### Progress not showing
- Ensure gaze data is being passed to components
- Check Y threshold (buttons at Y > 0.55)
- Verify button indices match (4-5 for actions, 6-7 for solution)

## Future Enhancements

Possible improvements:
- Adjustable dwell time per user preference
- Calibration screen for personalized accuracy
- Visual feedback during calibration
- Support for more button layouts
- Accessibility settings panel
