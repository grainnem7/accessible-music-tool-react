import React, { useRef, useState, useEffect } from 'react';
import ReactWebcam from 'react-webcam';
import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs';
import './WebcamCapture.css';
import { AzureVisionService } from '../services/AzureVisionService';

// TypeScript casting to fix type issues
const Webcam = ReactWebcam as any;

interface WebcamCaptureProps {
  onPoseDetected: (poses: poseDetection.Pose[]) => void;
  showDebugInfo?: boolean;
  highlightIntentional?: boolean;
  intentionalKeypoints?: string[];
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
  // Additional connections from the reference code
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
  intentionalKeypoints = []
}) => {
  const webcamRef = useRef<any>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [detector, setDetector] = useState<poseDetection.PoseDetector | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [modelType, setModelType] = useState<'BlazePose' | 'MoveNet'>('MoveNet'); // Start with MoveNet for faster loading
  const [showLabels, setShowLabels] = useState(true);
  const [poseHistory, setPoseHistory] = useState<poseDetection.Pose[]>([]);
  const [velocities, setVelocities] = useState<Record<string, number>>({});
  const [containerSize, setContainerSize] = useState({ width: 640, height: 480 });
  const [useAzureVision, setUseAzureVision] = useState<boolean>(false);
  const [azureVisionService] = useState(new AzureVisionService());
  
  // Initialize TensorFlow.js
  useEffect(() => {
    const setupTensorflow = async () => {
      try {
        console.log('Initializing TensorFlow.js...');
        
        // Explicitly initialize TensorFlow
        await tf.ready();
        
        // Set a specific backend (webgl is generally more stable than webgpu)
        if (tf.getBackend() !== 'webgl') {
          await tf.setBackend('webgl');
        }
        
        // Check if backend is actually set
        const backend = tf.getBackend();
        if (backend) {
          console.log('TensorFlow.js initialized with backend:', backend);
        } else {
          // Try another backend if webgl fails
          await tf.setBackend('cpu');
          console.log('Fallback to CPU backend');
        }
        
        // Configure memory management for WebGL
        try {
          // Use publicAPI to set WebGL texture threshold
          if (backend === 'webgl') {
            tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0);
          }
        } catch (e) {
          console.warn('Could not configure WebGL texture management', e);
        }
      } catch (error) {
        console.error('TensorFlow initialization error:', error);
        alert('Error initializing TensorFlow.js. Please try reloading the page.');
      }
    };
    
    setupTensorflow();
    
    // Cleanup
    return () => {
      // Dispose any lingering tensors
      tf.disposeVariables();
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

  // Initialize the pose detector
  useEffect(() => {
    const initializeDetector = async () => {
      try {
        setIsModelLoading(true);
        
        // Make sure TensorFlow is ready
        await tf.ready();
        
        // Dispose old detector if exists
        if (detector) {
          detector.dispose();
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
  }, [modelType]);

  // Run pose detection
  useEffect(() => {
    if (isModelLoading) return;
    if (!detector && !useAzureVision) return;
    
    let animationFrameId: number;
    
    const detectPose = async () => {
      if (
        webcamRef.current &&
        webcamRef.current.video &&
        webcamRef.current.video.readyState === 4
      ) {
        try {
          const video = webcamRef.current.video;
          let poses: poseDetection.Pose[] = [];
          
          if (useAzureVision) {
            // Use Azure Computer Vision
            try {
              const videoBlob = await azureVisionService.captureVideoFrame(video);
              const azurePoseResult = await azureVisionService.detectPose(videoBlob);
              
              if (azurePoseResult && azurePoseResult.keypoints.length > 0) {
                // Convert Azure keypoints to TensorFlow.js format
                poses = [{
                  keypoints: azurePoseResult.keypoints.map(kp => ({
                    name: kp.name,
                    x: kp.position.x,
                    y: kp.position.y,
                    score: kp.confidence
                  })),
                  score: 1.0  // Overall pose confidence
                }];
              }
            } catch (azureError) {
              console.error('Azure Vision error, falling back to TensorFlow:', azureError);
              if (detector) {
                poses = await detector.estimatePoses(video, {
                  flipHorizontal: false,
                  maxPoses: 1
                });
              }
            }
          } else if (detector) {
            // Use TensorFlow.js
            poses = await detector.estimatePoses(video, {
              flipHorizontal: false,
              maxPoses: 1
            });
          }
          
          if (poses && poses.length > 0) {
            // Ensure canvas dimensions match the video
            if (canvasRef.current) {
              const videoWidth = video.videoWidth;
              const videoHeight = video.videoHeight;
              
              // Set canvas size directly
              canvasRef.current.width = videoWidth;
              canvasRef.current.height = videoHeight;
              
              // Draw poses on canvas
              drawPoses(poses, videoWidth, videoHeight, intentionalKeypoints);
            }
            
            // Keep a history of poses for velocity calculation
            setPoseHistory(prev => {
              const newHistory = [...prev, poses[0]];
              if (newHistory.length > 30) { // Keep last 30 frames (~1 second)
                return newHistory.slice(-30);
              }
              return newHistory;
            });
            
            // Calculate velocities for keypoints
            if (poseHistory.length >= 2) {
              const prevPose = poseHistory[poseHistory.length - 1];
              const currentPose = poses[0];
              
              const newVelocities: Record<string, number> = {};
              
              currentPose.keypoints.forEach(kp => {
                if (kp.name) {
                  const prevKp = prevPose.keypoints.find(p => p.name === kp.name);
                  if (prevKp && typeof prevKp.x === 'number' && typeof prevKp.y === 'number' && 
                      typeof kp.x === 'number' && typeof kp.y === 'number') {
                    const dx = kp.x - prevKp.x;
                    const dy = kp.y - prevKp.y;
                    newVelocities[kp.name] = Math.sqrt(dx*dx + dy*dy);
                  }
                }
              });
              
              setVelocities(newVelocities);
            }
            
            // Send poses to parent component
            onPoseDetected(poses);
          }
        } catch (error) {
          console.error('Error detecting poses:', error);
        }
      }
      
      // Continue detection loop
      animationFrameId = requestAnimationFrame(detectPose);
    };
    
    detectPose();
    
    // Clean up
    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [detector, isModelLoading, onPoseDetected, intentionalKeypoints, poseHistory, useAzureVision, azureVisionService]);

  // Draw the detected poses on the canvas
  const drawPoses = (
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
          
          // Only add text labels if enabled
          if (showLabels && keypoint.name) {
            // Background for text (for better visibility)
            const labelText = `${keypoint.name}: ${Math.round(keypoint.score * 100)}%${isIntentional ? ' (I)' : ''}`;
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
  };

  // Switch between pose detection models
  const switchModelType = () => {
    setModelType(prevType => prevType === 'BlazePose' ? 'MoveNet' : 'BlazePose');
  };

  // Toggle label visibility
  const toggleLabels = () => {
    setShowLabels(prevShowLabels => !prevShowLabels);
  };
  
  // Toggle Azure Vision
  const toggleAzureVision = () => {
    setUseAzureVision(prev => !prev);
  };

  // Render debug info panel
  const renderDebugPanel = () => {
    if (!showDebugInfo || poseHistory.length === 0) return null;
    
    const latestPose = poseHistory[poseHistory.length - 1];
    
    return (
      <div style={{
        position: 'absolute',
        top: 10,
        right: 10,
        background: 'rgba(0,0,0,0.7)',
        color: 'white',
        padding: 10,
        borderRadius: 5,
        maxHeight: 300,
        overflow: 'auto',
        fontSize: 12,
        zIndex: 30
      }}>
        <h3 style={{ margin: '0 0 8px 0' }}>Keypoints & Velocities</h3>
        {latestPose.keypoints
          .filter(kp => kp.score && kp.score > 0.5 && kp.name)
          .map((kp, i) => {
            const isIntentional = intentionalKeypoints.includes(kp.name || '');
            return (
              <div key={i} style={{ 
                display: 'flex', 
                justifyContent: 'space-between',
                marginBottom: '4px',
                borderBottom: '1px solid rgba(255,255,255,0.2)',
                backgroundColor: isIntentional ? 'rgba(255, 215, 0, 0.3)' : 'transparent'
              }}>
                <span>{kp.name}: {Math.round((kp.score || 0) * 100)}%</span>
                <span>Vel: {velocities[kp.name || '']?.toFixed(1) || 'N/A'}</span>
              </div>
            );
          })}
      </div>
    );
  };

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
      
      {/* Debug panel */}
      {renderDebugPanel()}
      
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
          onClick={toggleAzureVision}
          className="azure-vision-button"
          style={{
            backgroundColor: useAzureVision ? 'rgba(0,120,212,0.8)' : 'rgba(0,0,0,0.7)',
            color: 'white',
            border: 'none',
            padding: '8px 12px',
            borderRadius: '5px',
            cursor: 'pointer',
            fontSize: '14px'
          }}
        >
          {useAzureVision ? 'Use TensorFlow.js' : 'Use Azure Vision'}
        </button>
        
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
        >
          Switch to {modelType === 'BlazePose' ? 'MoveNet' : 'BlazePose'} Model
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

export default WebcamCapture;