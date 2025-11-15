import { useEffect, useRef, useState } from 'react';

interface GazeData {
  card_index: number;
  progress: number;
  gaze_x: number;
  gaze_y: number;
  face_detected: boolean;
}

export function useEyeTracking(
  enabled: boolean,
  onSelect: (index: number) => void
) {
  const [gazeData, setGazeData] = useState<GazeData | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const onSelectRef = useRef(onSelect);

  // Update the ref when onSelect changes
  useEffect(() => {
    onSelectRef.current = onSelect;
  }, [onSelect]);

  useEffect(() => {
    if (!enabled) {
      // Cleanup if disabled
      if (wsRef.current && wsRef.current.readyState !== WebSocket.CLOSED) {
        console.log('🔌 Closing WebSocket connection');
        wsRef.current.close();
        wsRef.current = null;
      }
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach(t => t.stop());
        videoRef.current.srcObject = null;
      }
      setConnected(false);
      setGazeData(null);
      return;
    }

    // Prevent multiple connections
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      console.log('⚠️ WebSocket already connected, skipping');
      return;
    }

    // Connect to WebSocket
    console.log('🔌 Connecting to eye tracking service...');
    const ws = new WebSocket('ws://localhost:8002/ws/eye-tracking');
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('✅ Eye tracking connected');
      setConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === 'gaze') {
          setGazeData({
            card_index: data.card_index,
            progress: data.progress,
            gaze_x: data.gaze_x,
            gaze_y: data.gaze_y,
            face_detected: data.face_detected
          });
        } else if (data.type === 'selection') {
          console.log('👁️ Eye tracking selection:', data.card_index);
          onSelectRef.current(data.card_index);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('❌ WebSocket error:', error);
      setConnected(false);
    };

    ws.onclose = () => {
      console.log('🔌 Eye tracking disconnected');
      setConnected(false);
    };

    // Start camera
    navigator.mediaDevices.getUserMedia({ 
      video: { 
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: 'user'
      } 
    })
      .then(stream => {
        if (videoRef.current && !videoRef.current.srcObject) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(err => {
            console.log('Video play interrupted (normal during React re-renders):', err.message);
          });
        }
      })
      .catch(error => {
        console.error('❌ Camera access error:', error);
      });

    // Send frames to backend
    intervalRef.current = setInterval(() => {
      if (
        videoRef.current &&
        canvasRef.current &&
        ws.readyState === WebSocket.OPEN &&
        videoRef.current.readyState === videoRef.current.HAVE_ENOUGH_DATA
      ) {
        const canvas = canvasRef.current;
        const video = videoRef.current;
        const ctx = canvas.getContext('2d');

        if (!ctx) return;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);

        try {
          const frameData = canvas.toDataURL('image/jpeg', 0.8);
          ws.send(JSON.stringify({ type: 'frame', frame: frameData }));
        } catch (error) {
          console.error('Error sending frame:', error);
        }
      }
    }, 33); // ~30 FPS

    // Cleanup
    return () => {
      console.log('🧹 Cleaning up eye tracking...');
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close();
      }
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach(t => t.stop());
        videoRef.current.srcObject = null;
      }
    };
  }, [enabled]); // Removed onSelect from dependencies

  return { gazeData, videoRef, canvasRef, connected };
}
