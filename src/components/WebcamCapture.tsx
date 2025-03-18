import React, { useRef, useState, useEffect, useCallback } from 'react';
import ReactWebcam from 'react-webcam';
import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs';
import './WebcamCapture.css';
import { performanceMonitor } from '../utils/Helpers/PerformanceMonitor';

// Use 'any' to bypass TypeScript errors for now
const Webcam: any = ReactWebcam;

interface WebcamCaptureProps {
  onPoseDetected: (poses: poseDetection.Pose[]) => void;
  showDebugInfo?: boolean;
  highlightIntentional?: boolean;
  intentionalKeypoints?: string[];
  frameRate?: number; // Target frame rate, default 30fps
}

// Skeleton connections mapping for BlazePose
const POSE_CONNECTIONS = [
  ['nose', 'left_eye'],
  ['nose', 'right_eye'],
  ['left_eye', 'left_ear'],
  ['right_eye', 'right_ear'],
  ['left_shoulder', 'right_shoulder'],
  ['left_shoulder', 'left_elbow'],
  ['right_shoulder', 'right_elbow'],
  ['left_elbow', 'left_wrist'],
  ['right_elbow', 'right_wrist'],
  ['left_wrist', 'left_thumb'],
  ['left_wrist', 'left_index'],
  ['left_wrist', 'left_pinky'],
  ['right_wrist', 'right_thumb'],
  ['right_wrist', 'right_index'],
  ['right_wrist', 'right_pinky'],
  ['nose', 'left_shoulder'], 
  ['nose', 'right_shoulder'],
  ['left_shoulder', 'left_hip'], 
  ['right_shoulder', 'right_hip'],
  ['left_hip', 'right_hip'],
  ['left_hip', 'left_knee'], 
  ['right_hip', 'right_knee'],
  ['left_knee', 'left_ankle'], 
  ['right_knee', 'right_ankle']
];

const WebcamCapture: React.FC<WebcamCaptureProps> = ({ 
  onPoseDetected,
  showDebugInfo = true,
  highlightIntentional = false,
  intentionalKeypoints = [],
  frameRate = 30
}) => {
  const webcamRef = useRef<any>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [detector, setDetector] = useState<poseDetection.PoseDetector | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [modelType, setModelType] = useState<'BlazePose' | 'MoveNet'>('MoveNet'); // Start with MoveNet for faster loading
  const [showLabels, setShowLabels] = useState(true);
  const [containerSize, setContainerSize] = useState({ width: 640, height: 480 });
  const [frameCount, setFrameCount] = useState(0);
  const [fps, setFps] = useState(0);
  const [memoryStats, setMemoryStats] = useState<any>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const animationFrameRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef<number>(0);
  const fpsCounterRef = useRef<number>(0);
  const fpsTimerRef = useRef<number>(0);
  
  // Enable performance monitoring
  useEffect(() => {
    performanceMonitor.setEnabled(true);
    performanceMonitor.setWarningThreshold('poseDetection', 33); // 33ms = 30fps
    
    return () => {
      performanceMonitor.setEnabled(false);
    };
  }, []);
  
  // Track container size
  useEffect(() => {
    if (!containerRef.current) return;
    
    const updateContainerSize = () => {
      if (containerRef.current) {
        const { width, height } = containerRef.current.getBoundingClientRect();
        setContainerSize({ width, height });
      }
    };
    
    // Initial size
    updateContainerSize();
    
    // Listen for resize
    window.addEventListener('resize', updateContainerSize);
    
    return () => {
      window.removeEventListener('resize', updateContainerSize);
    };
  }, []);

  // Initialize TensorFlow.js
  useEffect(() => {
    const setupTensorflow = async () => {
      try {
        console.log('Initializing TensorFlow.js...');
        
        await tf.ready();
        
        // Try to set WebGL backend first
        try {
          await tf.setBackend('webgl');
          console.log('Using WebGL backend');
          
          // Optimize WebGL performance
          tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0);
          tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
          tf.env().set('WEBGL_FLUSH_THRESHOLD', 1);
        } catch (e) {
          console.warn('WebGL backend failed, trying CPU', e);
          await tf.setBackend('cpu');
          console.log('Using CPU backend (slower)');
        }
      } catch (error) {
        console.error('TensorFlow initialization error:', error);
      }
    };
    
    setupTensorflow();
    
    // Start FPS measuring
    fpsTimerRef.current = window.setInterval(() => {
      setFps(fpsCounterRef.current);
      fpsCounterRef.current = 0;
    }, 1000);
    
    // Memory usage reporting
    const memoryTimer = window.setInterval(() => {
      try {
        const memInfo = tf.memory();
        setMemoryStats({
          numTensors: memInfo.numTensors,
          numBytes: (memInfo.numBytes / (1024 * 1024)).toFixed(2) + ' MB'
        });
      } catch (error) {
        console.warn('Error getting memory info:', error);
      }
    }, 5000);
    
    // Cleanup
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      
      clearInterval(fpsTimerRef.current);
      clearInterval(memoryTimer);
      
      // Clean up tensors
      tf.disposeVariables();
    };
  }, []);

  // Initialize the pose detector
  useEffect(() => {
    const initializeDetector = async () => {
      try {
        setIsModelLoading(true);
        
        // Make sure TensorFlow is ready
        await tf.ready();
        
        // Dispose old detector if exists
        if (detector) {
          await detector.dispose();
        }
        
        let newDetector;
        
        if (modelType === 'BlazePose') {
          // Create BlazePose detector
          const model = poseDetection.SupportedModels.BlazePose;
          const detectorConfig = {
            runtime: 'tfjs',
            enableSmoothing: true,
            modelType: 'lite' // Use 'lite' for faster loading
          };
          newDetector = await poseDetection.createDetector(model, detectorConfig);
          console.log('BlazePose detector created');
        } else {
          // Create MoveNet detector (faster and more reliable for our use case)
          const model = poseDetection.SupportedModels.MoveNet;
          const detectorConfig = {
            modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
            enableSmoothing: true,
            minPoseScore: 0.25
          };
          newDetector = await poseDetection.createDetector(model, detectorConfig);
          console.log('MoveNet detector created');
        }
        
        setDetector(newDetector);
        setIsModelLoading(false);
        
        console.log(`Pose detection model loaded: ${modelType}`);
      } catch (error) {
        console.error('Error initializing pose detector:', error);
        setIsModelLoading(false);
      }
    };

    initializeDetector();
    
    // Clean up function
    return () => {
      if (detector) {
        detector.dispose();
      }
    };
  }, [modelType, detector]);

  // Main pose detection loop
  useEffect(() => {
    if (!detector || isModelLoading) return;
    
    const detectPose = async (timestamp: number) => {
      if (
        webcamRef.current &&
        webcamRef.current.video &&
        webcamRef.current.video.readyState === 4
      ) {
        // FPS throttling - target the specified frame rate
        const targetFrameInterval = 1000 / frameRate;
        
        // Skip frames to maintain target frame rate
        const elapsed = timestamp - lastFrameTimeRef.current;
        if (elapsed < targetFrameInterval) {
          animationFrameRef.current = requestAnimationFrame(detectPose);
          return;
        }
        
        // Update last frame time
        lastFrameTimeRef.current = timestamp;
        
        try {
          // Skip if already processing a frame (prevents buildup of tasks)
          if (isProcessing) {
            animationFrameRef.current = requestAnimationFrame(detectPose);
            return;
          }
          
          setIsProcessing(true);
          
          // Start performance monitoring
          performanceMonitor.start('poseDetection');
          
          const video = webcamRef.current.video;
          
          // Detect poses
          const poses = await detector.estimatePoses(video, {
            flipHorizontal: false,
            maxPoses: 1
          });
          
          if (poses && poses.length > 0) {
            // Update canvas in a more efficient way
            requestAnimationFrame(() => {
              if (canvasRef.current) {
                const videoWidth = video.videoWidth;
                const videoHeight = video.videoHeight;
                
                // Set canvas size directly
                canvasRef.current.width = videoWidth;
                canvasRef.current.height = videoHeight;
                
                // Draw poses on canvas
                drawPoses(poses, videoWidth, videoHeight, intentionalKeypoints);
              }
            });
            
            // Notify parent component
            onPoseDetected(poses);
          }
          
          // Count frame for FPS calculation
          fpsCounterRef.current += 1;
          setFrameCount(prev => prev + 1);
          
          // End performance monitoring
          performanceMonitor.end('poseDetection');
          
          // Force garbage collection periodically
          if (frameCount % 100 === 0) {
            tf.tidy(() => {}); // Clean up unused tensors
          }
        } catch (error) {
          console.error('Error detecting poses:', error);
        } finally {
          setIsProcessing(false);
        }
      }
      
      // Continue detection loop
      animationFrameRef.current = requestAnimationFrame(detectPose);
    };
    
    // Start detection loop
    animationFrameRef.current = requestAnimationFrame(detectPose);
    
    // Clean up
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [detector, isModelLoading, onPoseDetected, intentionalKeypoints, frameCount, frameRate]);

  // Draw the detected poses on the canvas
  const drawPoses = useCallback((
    poses: poseDetection.Pose[], 
    width: number, 
    height: number,
    intentionalKeypoints: string[] = []
  ) => {
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Draw each pose
    poses.forEach((pose) => {
      // Create map of keypoints by name for easy lookup
      const keypointMap = new Map();
      pose.keypoints.forEach(keypoint => {
        if (keypoint.name) {
          keypointMap.set(keypoint.name, {
            ...keypoint,
            isIntentional: intentionalKeypoints.includes(keypoint.name)
          });
        }
      });
      
      // Draw skeleton (connecting lines between keypoints)
      ctx.lineWidth = 2;
      
      POSE_CONNECTIONS.forEach(([start, end]) => {
        const startPoint = keypointMap.get(start);
        const endPoint = keypointMap.get(end);
        
        if (startPoint && endPoint && 
            startPoint.score && endPoint.score &&
            startPoint.score > 0.3 && endPoint.score > 0.3) {
          
          // Determine if both points are intentional (for highlighting)
          const isConnectionIntentional = highlightIntentional && 
                                         startPoint.isIntentional && 
                                         endPoint.isIntentional;
          
          ctx.beginPath();
          ctx.moveTo(startPoint.x, startPoint.y);
          ctx.lineTo(endPoint.x, endPoint.y);
          
          if (isConnectionIntentional) {
            ctx.strokeStyle = 'rgba(255, 215, 0, 0.8)'; // Gold for intentional
            ctx.lineWidth = 4;
          } else {
            ctx.strokeStyle = 'rgba(0, 255, 0, 0.7)'; // Green for normal
            ctx.lineWidth = 2;
          }
          
          ctx.stroke();
        }
      });
      
      // Draw keypoints
      pose.keypoints.forEach((keypoint) => {
        if (keypoint.score && keypoint.score > 0.3) { // Only draw high-confidence keypoints
          const isIntentional = intentionalKeypoints.includes(keypoint.name || '');
          
          // Draw keypoint circle
          ctx.beginPath();
          ctx.arc(keypoint.x, keypoint.y, isIntentional ? 7 : 5, 0, 2 * Math.PI);
          
          if (isIntentional) {
            ctx.fillStyle = '#FFD700'; // Gold for intentional
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 2;
            ctx.stroke();
          } else {
            if (keypoint.score > 0.7) {
              ctx.fillStyle = '#00FF00'; // Green for high confidence
            } else if (keypoint.score > 0.5) {
              ctx.fillStyle = '#FFFF00'; // Yellow for medium confidence
            } else {
              ctx.fillStyle = '#FF0000'; // Red for low confidence
            }
          }
          
          ctx.fill();
          
          // Only add text labels if enabled and not too many to avoid clutter
          if (showLabels && keypoint.name && keypoint.score > 0.5) {
            // Background for text (for better visibility)
            const labelText = isIntentional 
              ? `${keypoint.name} ⭐`  // Star for intentional
              : keypoint.name;
            
            const textWidth = ctx.measureText(labelText).width;
            
            ctx.fillStyle = isIntentional ? 'rgba(255, 215, 0, 0.7)' : 'rgba(0, 0, 0, 0.7)';
            ctx.fillRect(
              keypoint.x + 10, 
              keypoint.y - 10, 
              textWidth + 6, 
              20
            );
            
            // Draw text
            ctx.font = '12px Arial';
            ctx.fillStyle = isIntentional ? 'black' : 'white';
            ctx.fillText(labelText, keypoint.x + 13, keypoint.y + 5);
          }
        }
      });
    });
  }, [highlightIntentional, showLabels]);

  // Switch between pose detection models
  const switchModelType = useCallback(() => {
    setModelType(prevType => prevType === 'BlazePose' ? 'MoveNet' : 'BlazePose');
  }, []);

  // Toggle label visibility
  const toggleLabels = useCallback(() => {
    setShowLabels(prevShowLabels => !prevShowLabels);
  }, []);

  return (
    <div 
      ref={containerRef}
      className="webcam-container"
      style={{ 
        position: 'relative',
        width: '100%',
        maxWidth: '640px',
        margin: '0 auto',
        overflow: 'hidden',
        borderRadius: '10px',
        boxShadow: '0 4px 15px rgba(0, 0, 0, 0.3)'
      }}
    >
      {isModelLoading && (
        <div className="loading-indicator" style={{ 
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          backgroundColor: 'rgba(0,0,0,0.7)',
          color: 'white',
          padding: '10px 20px',
          borderRadius: '5px',
          zIndex: 50
        }}>
          Loading pose detection model...
        </div>
      )}
      
      <Webcam
        ref={webcamRef}
        audio={false}
        screenshotFormat="image/jpeg"
        width={640}
        height={480}
        videoConstraints={{
          width: 640,
          height: 480,
          facingMode: "user"
        }}
        className="webcam-video"
        style={{
          width: '100%',
          height: 'auto',
          filter: isModelLoading ? 'blur(4px)' : 'none'
        }}
      />
      
      <canvas
        ref={canvasRef}
        className="webcam-canvas"
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%'
        }}
      />
      
      {/* Performance stats */}
      {showDebugInfo && (
        <div style={{ 
          position: 'absolute',
          top: '10px',
          right: '10px',
          background: 'rgba(0,0,0,0.7)',
          color: 'white',
          padding: '10px',
          borderRadius: '5px',
          fontSize: '12px',
          zIndex: 40
        }}>
          <div>FPS: {fps}</div>
          {memoryStats && (
            <>
              <div>Tensors: {memoryStats.numTensors}</div>
              <div>Memory: {memoryStats.numBytes}</div>
            </>
          )}
          <div>Model: {modelType}</div>
        </div>
      )}
      
      {/* Controls */}
      <div style={{ 
        position: 'absolute',
        bottom: '10px',
        right: '10px',
        display: 'flex',
        gap: '10px',
        zIndex: 40
      }}>
        <button 
          onClick={switchModelType}
          className="model-switch-button"
          style={{
            backgroundColor: 'rgba(0,0,0,0.7)',
            color: 'white',
            border: 'none',
            padding: '8px 12px',
            borderRadius: '5px',
            cursor: 'pointer',
            fontSize: '14px'
          }}
          disabled={isModelLoading}
        >
          Switch Model
        </button>
        
        <button 
          onClick={toggleLabels}
          className="labels-button"
          style={{
            backgroundColor: 'rgba(0,0,0,0.7)',
            color: 'white',
            border: 'none',
            padding: '8px 12px',
            borderRadius: '5px',
            cursor: 'pointer',
            fontSize: '14px'
          }}
        >
          {showLabels ? 'Hide' : 'Show'} Labels
        </button>
      </div>
    </div>
  );
};

export default React.memo(WebcamCapture);