import React, { useRef, useState, useEffect } from 'react';
import ReactWebcam from 'react-webcam';
import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs';

// TypeScript type assertion to fix type issues
const Webcam = ReactWebcam as any;

interface WebcamCaptureProps {
  onPoseDetected: (poses: poseDetection.Pose[]) => void;
  showDebugInfo?: boolean;
  highlightIntentional?: boolean;
  intentionalKeypoints?: string[];
}

// Extended keypoint for DOM positioning
interface ExtendedKeypoint extends poseDetection.Keypoint {
  id: string;
  isIntentional?: boolean;
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
  ['right_wrist', 'right_pinky']
];

const WebcamCapture: React.FC<WebcamCaptureProps> = ({ 
  onPoseDetected,
  showDebugInfo = true,
  highlightIntentional = false,
  intentionalKeypoints = []
}) => {
  const webcamRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [detector, setDetector] = useState<poseDetection.PoseDetector | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [modelType, setModelType] = useState<'BlazePose' | 'MoveNet'>('BlazePose');
  const [showLabels, setShowLabels] = useState(true);
  const [detectedKeypoints, setDetectedKeypoints] = useState<ExtendedKeypoint[]>([]);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const [poseHistory, setPoseHistory] = useState<poseDetection.Pose[]>([]);
  const [velocities, setVelocities] = useState<Record<string, number>>({});
  
  // Initialize TensorFlow.js
  useEffect(() => {
    const setupTensorflow = async () => {
      try {
        // Explicitly initialize TensorFlow
        await tf.ready();
        
        // Set a specific backend (webgl is generally more stable than webgpu)
        if (tf.getBackend() !== 'webgl') {
          await tf.setBackend('webgl');
        }
        
        console.log('TensorFlow.js initialized with backend:', tf.getBackend());
      } catch (error) {
        console.error('TensorFlow initialization error:', error);
      }
    };
    
    setupTensorflow();
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
          // Create BlazePose detector (better for hands and upper body detail)
          const model = poseDetection.SupportedModels.BlazePose;
          const detectorConfig = {
            runtime: 'tfjs',
            enableSmoothing: true,
            modelType: 'full' // 'lite', 'full', or 'heavy'
          };
          newDetector = await poseDetection.createDetector(model, detectorConfig);
          console.log('BlazePose detector created');
        } else {
          // Create MoveNet detector
          const model = poseDetection.SupportedModels.MoveNet;
          const detectorConfig = {
            modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER,
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
    if (!detector || isModelLoading) return;
    
    let animationFrameId: number;
    
    const detectPose = async () => {
      if (
        webcamRef.current &&
        webcamRef.current.video &&
        webcamRef.current.video.readyState === 4
      ) {
        try {
          const video = webcamRef.current.video;
          
          // Detect poses
          const poses = await detector.estimatePoses(video, {
            flipHorizontal: false,
            maxPoses: 1
          });
          
          if (poses && poses.length > 0) {
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
            
            // Mark intentional keypoints if provided
            const markedKeypoints = poses[0].keypoints.map(kp => ({
              ...kp,
              id: `keypoint-${kp.name}-${Date.now()}`,
              isIntentional: intentionalKeypoints.includes(kp.name || '')
            })).filter(kp => kp.score && kp.score > 0.2);
            
            setDetectedKeypoints(markedKeypoints);
            
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
  }, [detector, isModelLoading, onPoseDetected, intentionalKeypoints]);

  // Find a keypoint by name
  const findKeypointByName = (name: string): ExtendedKeypoint | undefined => {
    return detectedKeypoints.find(kp => kp.name === name);
  };

  // Render skeleton lines connecting keypoints
  const renderSkeletonLines = () => {
    if (detectedKeypoints.length === 0) return null;
    
    // Get video dimensions to calculate correct positions
    const videoWidth = webcamRef.current?.video?.videoWidth || 800;
    const videoHeight = webcamRef.current?.video?.videoHeight || 600;
    
    // Calculate scale factors for positioning
    const scaleX = containerSize.width / videoWidth;
    const scaleY = containerSize.height / videoHeight;
    
    return (
      <svg 
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none'
        }}
      >
        {POSE_CONNECTIONS.map(([from, to], index) => {
          const fromKeypoint = findKeypointByName(from);
          const toKeypoint = findKeypointByName(to);
          
          if (!fromKeypoint || !toKeypoint) return null;
          
          // Determine if both points are intentional (for highlighting)
          const isConnectionIntentional = highlightIntentional && 
                                        fromKeypoint.isIntentional && 
                                        toKeypoint.isIntentional;
          
          return (
            <line
              key={`line-${from}-${to}-${index}`}
              x1={fromKeypoint.x ? fromKeypoint.x * scaleX : 0}
              y1={fromKeypoint.y ? fromKeypoint.y * scaleY : 0}
              x2={toKeypoint.x ? toKeypoint.x * scaleX : 0}
              y2={toKeypoint.y ? toKeypoint.y * scaleY : 0}
              stroke={isConnectionIntentional ? "rgba(255, 215, 0, 0.8)" : "rgba(0, 255, 0, 0.7)"}
              strokeWidth={isConnectionIntentional ? "4" : "2"}
            />
          );
        })}
      </svg>
    );
  };

  // Add this method to WebcamCapture component
const switchModelType = () => {
  setModelType(prevType => prevType === 'BlazePose' ? 'MoveNet' : 'BlazePose');
};

// Add this method to WebcamCapture component
const toggleLabels = () => {
  setShowLabels(prevShowLabels => !prevShowLabels);
};

  // Render keypoint labels
  const renderKeypointLabels = () => {
    if (!showLabels || detectedKeypoints.length === 0) return null;
    
    const getColor = (keypoint: ExtendedKeypoint) => {
      if (keypoint.isIntentional) return '#FFD700'; // Gold for intentional
      if (!keypoint.score) return '#FF0000';
      if (keypoint.score >= 0.7) return '#00FF00';
      if (keypoint.score >= 0.5) return '#FFFF00';
      return '#FF0000';
    };
    
    // Calculate scale factors for positioning
    const videoWidth = webcamRef.current?.video?.videoWidth || 800;
    const videoHeight = webcamRef.current?.video?.videoHeight || 600;
    const scaleX = containerSize.width / videoWidth;
    const scaleY = containerSize.height / videoHeight;
    
    return detectedKeypoints.map(keypoint => (
      <div 
        key={keypoint.id}
        style={{
          position: 'absolute',
          left: `${keypoint.x ? keypoint.x * scaleX : 0}px`,
          top: `${keypoint.y ? keypoint.y * scaleY : 0}px`,
          zIndex: 100,
          pointerEvents: 'none'
        }}
      >
        {/* Point marker */}
        <div style={{
          width: keypoint.isIntentional ? '20px' : '16px',
          height: keypoint.isIntentional ? '20px' : '16px',
          backgroundColor: getColor(keypoint),
          borderRadius: '50%',
          position: 'absolute',
          top: keypoint.isIntentional ? '-10px' : '-8px',
          left: keypoint.isIntentional ? '-10px' : '-8px',
          border: `2px solid ${keypoint.isIntentional ? 'white' : 'rgba(255,255,255,0.7)'}`,
          boxShadow: keypoint.isIntentional ? '0 0 10px gold' : 'none'
        }} />
        
        {/* Label */}
        {showLabels && (
          <div style={{
            position: 'absolute',
            left: '10px',
            top: '-10px',
            backgroundColor: keypoint.isIntentional ? 'rgba(255, 215, 0, 0.8)' : 'rgba(0, 0, 0, 0.7)',
            color: keypoint.isIntentional ? 'black' : 'white',
            padding: '2px 6px',
            borderRadius: '4px',
            fontSize: '12px',
            fontWeight: 'bold',
            whiteSpace: 'nowrap',
            textShadow: keypoint.isIntentional ? 'none' : '1px 1px 1px black',
            border: `1px solid ${getColor(keypoint)}`
          }}>
            {keypoint.name}: {Math.round((keypoint.score || 0) * 100)}%
            {keypoint.isIntentional && ' (Intentional)'}
          </div>
        )}
      </div>
    ));
  };

  // Render debug info panel
  const renderDebugPanel = () => {
    if (!showDebugInfo || detectedKeypoints.length === 0) return null;
    
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
        zIndex: 100
      }}>
        <h3 style={{ margin: '0 0 8px 0' }}>Keypoints & Velocities</h3>
        {detectedKeypoints.map((kp, i) => (
          <div key={i} style={{ 
            display: 'flex', 
            justifyContent: 'space-between',
            marginBottom: '4px',
            borderBottom: '1px solid rgba(255,255,255,0.2)',
            backgroundColor: kp.isIntentional ? 'rgba(255, 215, 0, 0.3)' : 'transparent'
          }}>
            <span>{kp.name}: {Math.round((kp.score || 0) * 100)}%</span>
            <span>Vel: {velocities[kp.name || '']?.toFixed(1) || 'N/A'}</span>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div 
      ref={containerRef}
      style={{ 
        position: 'relative',
        width: '800px',
        maxWidth: '100%',
        margin: '0 auto'
      }}
    >
      {isModelLoading && (
        <div style={{ 
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          backgroundColor: 'rgba(0,0,0,0.7)',
          color: 'white',
          padding: '10px 20px',
          borderRadius: '5px',
          zIndex: 10
        }}>
          Loading pose detection model...
        </div>
      )}
      
      <div style={{ position: 'relative', width: '100%' }}>
        <Webcam
          ref={webcamRef}
          audio={false}
          screenshotFormat="image/jpeg"
          width={800}
          height={600}
          videoConstraints={{
            width: 800,
            height: 600,
            facingMode: "user"
          }}
          style={{
            width: '100%',
            height: 'auto',
            filter: isModelLoading ? 'blur(4px)' : 'none'
          }}
        />
        
        {/* Skeleton lines */}
        {renderSkeletonLines()}
        
        {/* Keypoint labels */}
        {renderKeypointLabels()}
      </div>
      
      {/* Debug panel */}
      {renderDebugPanel()}
      
      {/* Controls */}
      <div style={{ 
        position: 'absolute',
        bottom: '10px',
        right: '10px',
        display: 'flex',
        gap: '10px'
      }}>
        <button 
          onClick={switchModelType}
          style={{
            backgroundColor: 'rgba(0,0,0,0.7)',
            color: 'white',
            border: 'none',
            padding: '8px 12px',
            borderRadius: '5px',
            cursor: 'pointer',
            fontSize: '14px',
            zIndex: 10
          }}
        >
          Switch to {modelType === 'BlazePose' ? 'MoveNet' : 'BlazePose'} Model
        </button>
        
        <button 
          onClick={toggleLabels}
          style={{
            backgroundColor: 'rgba(0,0,0,0.7)',
            color: 'white',
            border: 'none',
            padding: '8px 12px',
            borderRadius: '5px',
            cursor: 'pointer',
            fontSize: '14px',
            zIndex: 10
          }}
        >
          {showLabels ? 'Hide' : 'Show'} Labels
        </button>
      </div>
    </div>
  );
};

export default WebcamCapture;