import { useEffect, useRef, useState } from 'react';

interface GazeData {
  card_index: number;
  progress: number;
  gaze_x: number;
  gaze_y: number;
  face_detected: boolean;
}

type EyeTrackingMode = 'cards' | 'buttons' | 'solution' | 'cards_with_regenerate';

// Shared state across all hook instances
let sharedWs: WebSocket | null = null;
let sharedVideoRef: HTMLVideoElement | null = null;
let sharedCanvasRef: HTMLCanvasElement | null = null;
let sharedStream: MediaStream | null = null;
let sharedInterval: NodeJS.Timeout | null = null;
let currentMode: EyeTrackingMode = 'cards';
let subscribers: Map<string, (data: GazeData) => void> = new Map();
let selectionHandlers: Map<string, (index: number) => void> = new Map();
let connectionCount = 0;

export function useSharedEyeTracking(
  enabled: boolean,
  onSelect: (index: number) => void,
  mode: EyeTrackingMode = 'cards',
  id: string // Unique identifier for this hook instance
) {
  const [gazeData, setGazeData] = useState<GazeData | null>(null);
  const [connected, setConnected] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const onSelectRef = useRef(onSelect);

  // Update the ref when onSelect changes
  useEffect(() => {
    onSelectRef.current = onSelect;
  }, [onSelect]);

  useEffect(() => {
    if (!enabled) {
      // Unsubscribe this instance
      subscribers.delete(id);
      selectionHandlers.delete(id);
      connectionCount--;

      // If no more subscribers, cleanup everything
      if (connectionCount <= 0) {
        connectionCount = 0;
        if (sharedInterval) {
          clearInterval(sharedInterval);
          sharedInterval = null;
        }
        if (sharedWs && sharedWs.readyState !== WebSocket.CLOSED) {
          console.log('🔌 Closing shared WebSocket connection');
          sharedWs.close();
          sharedWs = null;
        }
        if (sharedStream) {
          sharedStream.getTracks().forEach(t => t.stop());
          sharedStream = null;
        }
        if (sharedVideoRef) {
          sharedVideoRef.srcObject = null;
          sharedVideoRef = null;
        }
      }
      setConnected(false);
      setGazeData(null);
      return;
    }

    // Subscribe this instance
    connectionCount++;
    subscribers.set(id, setGazeData);
    selectionHandlers.set(id, onSelectRef.current);
    currentMode = mode;

    // Setup shared video and canvas refs
    if (videoRef.current && !sharedVideoRef) {
      sharedVideoRef = videoRef.current;
    }
    if (canvasRef.current && !sharedCanvasRef) {
      sharedCanvasRef = canvasRef.current;
    }

    // Only create connection if it doesn't exist
    if (!sharedWs || sharedWs.readyState === WebSocket.CLOSED) {
      console.log('🔌 Creating shared WebSocket connection...');
      const ws = new WebSocket('ws://localhost:8002/ws/eye-tracking');
      sharedWs = ws;

      ws.onopen = () => {
        console.log('✅ Shared eye tracking connected');
        subscribers.forEach(callback => {
          // Notify all subscribers about connection
        });
        setConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'gaze') {
            const gazeData = {
              card_index: data.card_index,
              progress: data.progress,
              gaze_x: data.gaze_x,
              gaze_y: data.gaze_y,
              face_detected: data.face_detected
            };
            // Notify all subscribers
            subscribers.forEach(callback => callback(gazeData));
          } else if (data.type === 'selection') {
            console.log('👁️ Eye tracking selection:', data.card_index);
            // Call the appropriate handler based on current mode
            selectionHandlers.forEach(handler => handler(data.card_index));
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        subscribers.forEach(() => setConnected(false));
      };

      ws.onclose = () => {
        console.log('🔌 Eye tracking disconnected');
        subscribers.forEach(() => setConnected(false));
      };

      // Start camera only once
      if (!sharedStream) {
        navigator.mediaDevices.getUserMedia({ 
          video: { 
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user'
          } 
        })
          .then(stream => {
            sharedStream = stream;
            if (sharedVideoRef && !sharedVideoRef.srcObject) {
              sharedVideoRef.srcObject = stream;
              sharedVideoRef.play().catch(err => {
                console.log('Video play interrupted:', err.message);
              });
            }
          })
          .catch(error => {
            console.error('❌ Camera access error:', error);
          });
      }

      // Send frames to backend
      if (!sharedInterval) {
        sharedInterval = setInterval(() => {
          if (
            sharedVideoRef &&
            sharedCanvasRef &&
            sharedWs &&
            sharedWs.readyState === WebSocket.OPEN &&
            sharedVideoRef.readyState === sharedVideoRef.HAVE_ENOUGH_DATA
          ) {
            const canvas = sharedCanvasRef;
            const video = sharedVideoRef;
            const ctx = canvas.getContext('2d');

            if (!ctx) return;

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0);

            try {
              const frameData = canvas.toDataURL('image/jpeg', 0.8);
              sharedWs.send(JSON.stringify({ 
                type: 'frame', 
                frame: frameData, 
                mode: currentMode 
              }));
            } catch (error) {
              console.error('Error sending frame:', error);
            }
          }
        }, 33); // ~30 FPS
      }
    } else {
      setConnected(true);
    }

    // Cleanup for this instance
    return () => {
      subscribers.delete(id);
      selectionHandlers.delete(id);
      connectionCount--;

      if (connectionCount <= 0) {
        connectionCount = 0;
        if (sharedInterval) {
          clearInterval(sharedInterval);
          sharedInterval = null;
        }
        if (sharedWs && (sharedWs.readyState === WebSocket.OPEN || sharedWs.readyState === WebSocket.CONNECTING)) {
          sharedWs.close();
          sharedWs = null;
        }
        if (sharedStream) {
          sharedStream.getTracks().forEach(t => t.stop());
          sharedStream = null;
        }
        if (sharedVideoRef) {
          sharedVideoRef.srcObject = null;
          sharedVideoRef = null;
        }
      }
    };
  }, [enabled, mode, id]);

  // Update current mode when it changes
  useEffect(() => {
    if (enabled) {
      currentMode = mode;
      console.log(`🔄 Switched eye tracking mode to: ${mode}`);
    }
  }, [mode, enabled]);

  return { gazeData, videoRef, canvasRef, connected };
}
