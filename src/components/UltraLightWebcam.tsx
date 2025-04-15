import React, { useRef, useState, useEffect } from 'react';
import ReactWebcam from 'react-webcam';
import './WebcamCapture.css';

// Use 'any' to bypass TypeScript errors
const Webcam: any = ReactWebcam;

interface UltraLightWebcamProps {
  onMotionDetected: (x: number, y: number, direction: string, velocity: number) => void;
  mirrored?: boolean; // Add mirroring option with default true
}

// This component uses simple motion detection with canvas pixel comparison
// No TensorFlow, no pose detection - just basic movement detection
const UltraLightWebcam: React.FC<UltraLightWebcamProps> = ({ 
  onMotionDetected,
  mirrored = true // Default to mirrored for natural interaction
}) => {
  const webcamRef = useRef<any>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const prevFrameRef = useRef<ImageData | null>(null);
  const animationRef = useRef<number | undefined>(undefined);
  const lastProcessTimeRef = useRef<number>(0);
  const [isReady, setIsReady] = useState(false);
  
  // Motion detection parameters
  const FPS = 10; // Very low frame rate to save CPU
  const FRAME_INTERVAL = 1000 / FPS;
  const MOTION_THRESHOLD = 30; // Pixel difference threshold
  const GRID_SIZE = 8; // Break the frame into an 8x8 grid for detection

  useEffect(() => {
    // Set up motion detection once webcam is ready
    const setupDetection = () => {
      if (
        webcamRef.current &&
        webcamRef.current.video &&
        webcamRef.current.video.readyState === 4 &&
        canvasRef.current
      ) {
        setIsReady(true);
        startDetection();
      } else {
        setTimeout(setupDetection, 100);
      }
    };
    
    setupDetection();
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);
  
  // Start the detection loop
  const startDetection = () => {
    const detectMotion = (time: number) => {
      // Throttle processing to the target FPS
      if (time - lastProcessTimeRef.current < FRAME_INTERVAL) {
        animationRef.current = requestAnimationFrame(detectMotion);
        return;
      }
      
      lastProcessTimeRef.current = time;
      
      try {
        const video = webcamRef.current?.video;
        const canvas = canvasRef.current;
        
        if (!video || !canvas) {
          animationRef.current = requestAnimationFrame(detectMotion);
          return;
        }
        
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          animationRef.current = requestAnimationFrame(detectMotion);
          return;
        }
        
        // Set canvas size if needed
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }
        
        // Draw current frame (mirrored if needed)
        ctx.save();
        if (mirrored) {
          // Mirror the image by scaling x by -1 and adjusting the drawing position
          ctx.translate(canvas.width, 0);
          ctx.scale(-1, 1);
        }
        ctx.drawImage(video, 0, 0);
        ctx.restore();
        
        // Get image data
        const currentFrame = ctx.getImageData(0, 0, canvas.width, canvas.height);
        
        // Skip if no previous frame
        if (!prevFrameRef.current) {
          prevFrameRef.current = currentFrame;
          animationRef.current = requestAnimationFrame(detectMotion);
          return;
        }
        
        // Calculate grid cell size
        const cellWidth = Math.floor(canvas.width / GRID_SIZE);
        const cellHeight = Math.floor(canvas.height / GRID_SIZE);
        
        // Track the cell with the most motion
        let maxDiffCell = { x: 0, y: 0, diff: 0 };
        
        // Check each grid cell for motion
        for (let gridY = 0; gridY < GRID_SIZE; gridY++) {
          for (let gridX = 0; gridX < GRID_SIZE; gridX++) {
            // Calculate pixel region to check in this cell
            const startX = gridX * cellWidth;
            const startY = gridY * cellHeight;
            const endX = Math.min(startX + cellWidth, canvas.width);
            const endY = Math.min(startY + cellHeight, canvas.height);
            
            let cellDiff = 0;
            let pixelCount = 0;
            
            // Sample pixels in this cell (skip pixels for performance)
            for (let y = startY; y < endY; y += 4) {
              for (let x = startX; x < endX; x += 4) {
                const i = (y * canvas.width + x) * 4;
                
                // Calculate color difference between frames
                const rDiff = Math.abs(currentFrame.data[i] - prevFrameRef.current.data[i]);
                const gDiff = Math.abs(currentFrame.data[i+1] - prevFrameRef.current.data[i+1]);
                const bDiff = Math.abs(currentFrame.data[i+2] - prevFrameRef.current.data[i+2]);
                
                // Average the color differences
                const pixelDiff = (rDiff + gDiff + bDiff) / 3;
                cellDiff += pixelDiff;
                pixelCount++;
              }
            }
            
            // Calculate average difference for this cell
            const avgDiff = pixelCount > 0 ? cellDiff / pixelCount : 0;
            
            // Track cell with most motion
            if (avgDiff > maxDiffCell.diff) {
              maxDiffCell = { x: gridX, y: gridY, diff: avgDiff };
            }
          }
        }
        
        // If motion exceeds threshold, report it
        if (maxDiffCell.diff > MOTION_THRESHOLD) {
          // Convert to screen coordinates
          const centerX = (maxDiffCell.x + 0.5) * cellWidth;
          const centerY = (maxDiffCell.y + 0.5) * cellHeight;
          
          // Determine direction based on position in the grid
          let direction = 'none';
          const centerGridX = GRID_SIZE / 2 - 0.5;
          const centerGridY = GRID_SIZE / 2 - 0.5;
          
          if (Math.abs(maxDiffCell.x - centerGridX) > Math.abs(maxDiffCell.y - centerGridY)) {
            // If mirrored, we need to flip left/right directions
            if (mirrored) {
              direction = maxDiffCell.x < centerGridX ? 'right' : 'left';
            } else {
              direction = maxDiffCell.x > centerGridX ? 'right' : 'left';
            }
          } else {
            direction = maxDiffCell.y > centerGridY ? 'down' : 'up';
          }
          
          // Scale the difference to a reasonable velocity value
          const velocity = Math.min(maxDiffCell.diff, 100);
          
          // Notify parent component - adjust X for mirroring if needed
          const adjustedX = mirrored ? canvas.width - centerX : centerX;
          
          onMotionDetected(adjustedX, centerY, direction, velocity);
          
          // Highlight the cell with motion (with mirroring adjustment)
          ctx.save();
          ctx.fillStyle = 'rgba(255, 0, 0, 0.3)';
          
          if (mirrored) {
            // We need to flip the highlighting to match the mirrored view
            const mirroredX = GRID_SIZE - 1 - maxDiffCell.x;
            ctx.fillRect(mirroredX * cellWidth, maxDiffCell.y * cellHeight, cellWidth, cellHeight);
          } else {
            ctx.fillRect(maxDiffCell.x * cellWidth, maxDiffCell.y * cellHeight, cellWidth, cellHeight);
          }
          ctx.restore();
        }
        
        // Store current frame as previous
        prevFrameRef.current = currentFrame;
      } catch (error) {
        console.error('Error in motion detection:', error);
      }
      
      // Continue the loop
      animationRef.current = requestAnimationFrame(detectMotion);
    };
    
    // Start the detection loop
    animationRef.current = requestAnimationFrame(detectMotion);
  };

  return (
    <div className="webcam-container">
      <Webcam
        ref={webcamRef}
        audio={false}
        screenshotFormat="image/jpeg"
        width={320} // Use smaller video size for better performance
        height={240}
        videoConstraints={{
          width: 320,
          height: 240,
          facingMode: "user"
        }}
        className="webcam-video"
        // Apply CSS mirroring to the video element
        style={{ transform: mirrored ? 'scaleX(-1)' : 'none' }}
      />
      
      <canvas
        ref={canvasRef}
        className="webcam-canvas"
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          visibility: 'hidden' // Hide the canvas, we just use it for processing
        }}
      />
      
      <div className="webcam-status">
        {isReady ? 'Ready' : 'Loading...'}
      </div>
    </div>
  );
};

export default UltraLightWebcam;