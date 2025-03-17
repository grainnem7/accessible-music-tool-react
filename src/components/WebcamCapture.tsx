import React, { useRef, useState, useEffect } from 'react';
import ReactWebcam from 'react-webcam';
import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs';

// TypeScript type assertion to fix type issues
const Webcam = ReactWebcam as any;

interface WebcamCaptureProps {
  onPoseDetected: (poses: poseDetection.Pose[]) => void;
  showDebugInfo?: boolean;
}

// Extended keypoint for DOM positioning
interface ExtendedKeypoint extends poseDetection.Keypoint {
  id: string;
}

const WebcamCapture: React.FC<WebcamCaptureProps> = ({ 
  onPoseDetected,
  showDebugInfo = true
}) => {
  const webcamRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [detector, setDetector] = useState<poseDetection.PoseDetector | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [modelType, setModelType] = useState<'BlazePose' | 'MoveNet'>('BlazePose');
  const [showLabels, setShowLabels] = useState(true);
  const [detectedKeypoints, setDetectedKeypoints] = useState<ExtendedKeypoint[]>([]);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  
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
        console.log('Container size:', width, height);
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

  // Run pose detection on regular intervals
  useEffect(() => {
    if (!detector || isModelLoading) return;
    
    let animationFrameId: number;
    
    const detectPose = async () => {
      if (
        webcamRef.current &&
        webcamRef.current.video &&
        webcamRef.current.video.readyState === 4 // Ready state 4 means video is ready
      ) {
        try {
          // Get video properties
          const video = webcamRef.current.video;
          
          // Detect poses
          const poses = await detector.estimatePoses(video, {
            flipHorizontal: false,
            maxPoses: 1
          });
          
          if (poses && poses.length > 0) {
            // Filter and prepare keypoints
            const validKeypoints = poses[0].keypoints
              .filter(kp => kp.score && kp.score > 0.2)
              .map(kp => ({
                ...kp,
                id: `keypoint-${kp.name}-${Date.now()}`
              }));
              
            setDetectedKeypoints(validKeypoints);
            
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
  }, [detector, isModelLoading, onPoseDetected, showLabels]);

  // Handle model type change
  const switchModelType = () => {
    const newType = modelType === 'BlazePose' ? 'MoveNet' : 'BlazePose';
    setModelType(newType);
  };

  // Toggle labels on/off
  const toggleLabels = () => {
    setShowLabels(!showLabels);
  };

  // Render DOM-based keypoint labels instead of canvas-based
  const renderKeypointLabels = () => {
    if (!showLabels || detectedKeypoints.length === 0) return null;
    
    const getColor = (score: number | undefined) => {
      if (!score) return '#FF0000';
      if (score >= 0.7) return '#00FF00';
      if (score >= 0.5) return '#FFFF00';
      return '#FF0000';
    };
    
    // Get video dimensions to calculate correct positions
    const videoWidth = webcamRef.current?.video?.videoWidth || 800;
    const videoHeight = webcamRef.current?.video?.videoHeight || 600;
    
    // Calculate scale factors for positioning
    const scaleX = containerSize.width / videoWidth;
    const scaleY = containerSize.height / videoHeight;
    
    return detectedKeypoints.map(keypoint => (
      <div 
        key={keypoint.id}
        style={{
          position: 'absolute',
          left: `${keypoint.x * scaleX}px`,
          top: `${keypoint.y * scaleY}px`,
          zIndex: 100,
          pointerEvents: 'none'
        }}
      >
        {/* Point marker */}
        <div style={{
          width: '16px',
          height: '16px',
          backgroundColor: getColor(keypoint.score),
          borderRadius: '50%',
          position: 'absolute',
          top: '-8px',
          left: '-8px',
          border: '2px solid white'
        }} />
        
        {/* Label */}
        <div style={{
          position: 'absolute',
          left: '10px',
          top: '-10px',
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          color: 'white',
          padding: '2px 6px',
          borderRadius: '4px',
          fontSize: '12px',
          fontWeight: 'bold',
          whiteSpace: 'nowrap',
          textShadow: '1px 1px 1px black',
          border: `1px solid ${getColor(keypoint.score)}`
        }}>
          {keypoint.name}: {Math.round((keypoint.score || 0) * 100)}%
        </div>
      </div>
    ));
  };

  // Render debug info panel with detected keypoints
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
        maxHeight: 200,
        overflow: 'auto',
        fontSize: 12,
        zIndex: 100
      }}>
        {detectedKeypoints.map((kp, i) => (
          <div key={i}>
            {kp.name}: {Math.round((kp.score || 0) * 100)}%
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
        
        {/* DOM-based keypoint labels */}
        {renderKeypointLabels()}
      </div>
      
      {/* Debug panel */}
      {renderDebugPanel()}
      
      {/* Sample counters */}
      <div style={{
        position: 'absolute',
        top: 10,
        left: 10,
        background: 'rgba(0,0,0,0.8)',
        padding: '8px',
        borderRadius: '5px',
        color: 'white',
        zIndex: 5
      }}>
        <div style={{ marginBottom: '5px' }}>
          <span style={{ 
            display: 'inline-block',
            padding: '4px 8px',
            marginRight: '10px',
            borderRadius: '3px',
            backgroundColor: '#4CAF50',
            fontWeight: 'bold'
          }}>
            Intentional:
          </span>
          <span style={{
            backgroundColor: 'rgba(255,255,255,0.2)',
            padding: '2px 8px',
            borderRadius: '10px',
            minWidth: '30px',
            textAlign: 'center',
            display: 'inline-block'
          }}>
            0
          </span>
        </div>
        <div>
          <span style={{ 
            display: 'inline-block',
            padding: '4px 8px',
            marginRight: '10px',
            borderRadius: '3px',
            backgroundColor: '#F44336',
            fontWeight: 'bold'
          }}>
            Unintentional:
          </span>
          <span style={{
            backgroundColor: 'rgba(255,255,255,0.2)',
            padding: '2px 8px',
            borderRadius: '10px',
            minWidth: '30px',
            textAlign: 'center',
            display: 'inline-block'
          }}>
            0
          </span>
        </div>
      </div>
      
      {/* Controls */}
      <button 
        onClick={switchModelType}
        style={{
          position: 'absolute',
          bottom: '10px',
          right: '10px',
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
          position: 'absolute',
          bottom: '10px',
          left: '10px',
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
  );
};

export default WebcamCapture;