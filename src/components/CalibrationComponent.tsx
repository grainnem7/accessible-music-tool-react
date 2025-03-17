import React, { useState, useEffect } from 'react';
import { MLIntentionDetector } from '../utils/MLIntentionDetector';
import WebcamCapture from './WebcamCapture';
import * as poseDetection from '@tensorflow-models/pose-detection';
import './CalibrationComponent.css';

interface CalibrationComponentProps {
  detector: MLIntentionDetector;
  onCalibrationComplete: (userId: string) => void;
}

const CalibrationComponent: React.FC<CalibrationComponentProps> = ({ 
  detector, 
  onCalibrationComplete 
}) => {
  const [calibrationStep, setCalibrationStep] = useState<number>(0);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [countdown, setCountdown] = useState<number>(0);
  const [userId, setUserId] = useState<string>('');
  const [modelStatus, setModelStatus] = useState({ 
    isModelTrained: false,
    calibrationSamples: 0,
    intentionalSamples: 0,
    unintentionalSamples: 0,
    isTfInitialized: false
  });
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainingProgress, setTrainingProgress] = useState<number>(0);
  const [showDebugInfo, setShowDebugInfo] = useState<boolean>(true); 
  
  // Timer for collecting samples
  useEffect(() => {
    let timer: NodeJS.Timeout;
    
    if (countdown > 0) {
      timer = setTimeout(() => setCountdown(countdown - 1), 1000);
    } else if (countdown === 0 && isRecording) {
      setIsRecording(false);
      // Move to next step when recording completes
      if (calibrationStep < 5) {
        setCalibrationStep(calibrationStep + 1);
      }
    }
    
    return () => {
      if (timer) clearTimeout(timer);
    };
  }, [countdown, isRecording, calibrationStep]);
  
  // Update model status
  useEffect(() => {
    const interval = setInterval(() => {
      if (detector) {
        setModelStatus(detector.getStatus());
      }
    }, 1000);
    
    return () => clearInterval(interval);
  }, [detector]);
  
  // Process poses for calibration
  const handlePoseDetected = (poses: poseDetection.Pose[]) => {
    if (isRecording) {
      // During calibration steps 1 & 2, we're collecting unintentional movements
      // During steps 3 & 4, we're collecting intentional movements
      const isIntentional = calibrationStep >= 3 && calibrationStep <= 4;
      detector.addCalibrationSample(isIntentional);
    }
  };
  
  // Start recording samples for the current step
  const startRecording = () => {
    setIsRecording(true);
    setCountdown(5); // Record for 5 seconds
  };
  
  // Train the model with collected samples
  const trainModel = async () => {
    setIsTraining(true);
    setTrainingProgress(0);
    
    // Set up progress tracking
    detector.setTrainingProgressCallback((progress: number) => {
      setTrainingProgress(Math.round(progress * 100));
    });
    
    const success = await detector.trainModel();
    setIsTraining(false);
    
    if (success) {
      // Save the model for this user
      if (userId) {
        await detector.saveModel(userId);
      }
      
      // Update status and move to completion
      setModelStatus(detector.getStatus());
      setCalibrationStep(6); // Complete
    } else {
      alert("Training failed. Please try again with more distinct movements.");
    }
  };
  
  // Complete calibration and notify parent
  const completeCalibration = () => {
    onCalibrationComplete(userId);
  };
  
  // Reset the calibration process
  const resetCalibration = () => {
    detector.clearCalibration();
    setCalibrationStep(0);
    setModelStatus(detector.getStatus());
  };
  
  // Render calibration instructions based on current step
  const renderInstructions = () => {
    switch (calibrationStep) {
      case 0: // Introduction
        return (
          <div className="calibration-intro">
            <h2>Welcome to Movement Calibration</h2>
            <p>This process will train the system to recognize your intentional movements.</p>
            <p>We'll record samples of both your unintentional and intentional movements.</p>
            <p>Please enter a unique user ID to save your calibration:</p>
            <input 
              type="text" 
              value={userId} 
              onChange={(e) => setUserId(e.target.value)} 
              placeholder="Your name or ID"
              className="user-id-input"
            />
            <button 
              onClick={() => setCalibrationStep(1)} 
              disabled={!userId}
              className="next-button"
            >
              Begin Calibration
            </button>
          </div>
        );
        
      case 1: // Record unintentional movements (resting)
        return (
          <div className="calibration-step">
            <h2>Step 1: Capture Resting Movements</h2>
            <p>First, we'll record your natural movements when at rest.</p>
            <p>Please sit or stand comfortably and <strong>avoid making any deliberate gestures</strong>.</p>
            <p>The system will record your natural movements and tremors.</p>
            
            {isRecording ? (
              <div className="recording-indicator">
                <div className="recording-dot"></div>
                <p>Recording... {countdown} seconds remaining</p>
              </div>
            ) : (
              <button onClick={startRecording} className="record-button">
                Start Recording
              </button>
            )}
          </div>
        );
        
      case 2: // Record more unintentional movements (casual)
        return (
          <div className="calibration-step">
            <h2>Step 2: Capture Casual Movements</h2>
            <p>Now, we'll record your casual, non-musical movements.</p>
            <p>Please move naturally as if you're adjusting your position or having a conversation.</p>
            <p>These movements should <strong>NOT</strong> be ones you want to trigger sounds.</p>
            
            {isRecording ? (
              <div className="recording-indicator">
                <div className="recording-dot"></div>
                <p>Recording... {countdown} seconds remaining</p>
              </div>
            ) : (
              <button onClick={startRecording} className="record-button">
                Start Recording
              </button>
            )}
          </div>
        );
        
      case 3: // Record intentional movements (simple)
        return (
          <div className="calibration-step">
            <h2>Step 3: Capture Intentional Movements</h2>
            <p>Now, let's record movements you want to use to create music.</p>
            <p>Please make clear, deliberate gestures such as:</p>
            <ul>
              <li>Raising your hand up and down</li>
              <li>Moving your hand side to side</li>
              <li>Making a specific gesture you'd like to use</li>
            </ul>
            <p>Make these movements <strong>deliberate and distinct</strong> from your casual movements.</p>
            
            {isRecording ? (
              <div className="recording-indicator">
                <div className="recording-dot"></div>
                <p>Recording... {countdown} seconds remaining</p>
              </div>
            ) : (
              <button onClick={startRecording} className="record-button">
                Start Recording
              </button>
            )}
          </div>
        );
        
      case 4: // Record more intentional movements (complex)
        return (
          <div className="calibration-step">
            <h2>Step 4: More Intentional Movements</h2>
            <p>Let's record more intentional gestures, focusing on different types.</p>
            <p>Try making movements with different:</p>
            <ul>
              <li>Speeds (fast and slow)</li>
              <li>Directions (up, down, left, right)</li>
              <li>Body parts (both hands, head movements)</li>
            </ul>
            
            {isRecording ? (
              <div className="recording-indicator">
                <div className="recording-dot"></div>
                <p>Recording... {countdown} seconds remaining</p>
              </div>
            ) : (
              <button onClick={startRecording} className="record-button">
                Start Recording
              </button>
            )}
          </div>
        );
        
      case 5: // Train the model
        return (
          <div className="calibration-step">
            <h2>Model Training</h2>
            <p>We've collected:</p>
            <ul>
              <li><strong>{modelStatus.intentionalSamples}</strong> intentional movement samples</li>
              <li><strong>{modelStatus.unintentionalSamples}</strong> unintentional movement samples</li>
            </ul>
            
            {isTraining ? (
              <div className="training-indicator">
                <p>Training your personalized model...</p>
                <div className="progress-container">
                  <div 
                    className="progress-bar-training" 
                    style={{width: `${trainingProgress}%`}}
                  ></div>
                </div>
                <p>{trainingProgress}% complete</p>
                <div className="loading-spinner"></div>
              </div>
            ) : (
              <>
                <button 
                  onClick={trainModel} 
                  className="train-button" 
                  disabled={modelStatus.calibrationSamples < 10}
                >
                  Train Model
                </button>
                <button onClick={resetCalibration} className="reset-button">
                  Reset & Start Over
                </button>
              </>
            )}
          </div>
        );
        
      case 6: // Calibration complete
        return (
          <div className="calibration-complete">
            <h2>Calibration Complete!</h2>
            <p>Your personalized movement model is trained and ready to use.</p>
            <p>The system will now recognize your intentional gestures and filter out unintentional movements.</p>
            <p>You can recalibrate anytime from the settings menu if needed.</p>
            
            <button onClick={completeCalibration} className="complete-button">
              Start Making Music
            </button>
            <button onClick={resetCalibration} className="reset-button">
              Recalibrate
            </button>
          </div>
        );
        
      default:
        return null;
    }
  };
  
  return (
    <div className="calibration-container">
      {/* Two-column layout for larger screens */}
      <div className="calibration-layout">
        <div className="webcam-section">
          <WebcamCapture 
            onPoseDetected={handlePoseDetected} 
            showDebugInfo={showDebugInfo}
          />
          
          {/* Sample count overlay directly on the webcam view */}
          <div className="sample-counters">
            <div className="sample-counter">
              <div className="sample-label intentional">Intentional:</div>
              <div className="sample-value">{modelStatus.intentionalSamples}</div>
            </div>
            <div className="sample-counter">
              <div className="sample-label unintentional">Unintentional:</div>
              <div className="sample-value">{modelStatus.unintentionalSamples}</div>
            </div>
          </div>
        </div>
        
        <div className="instructions-section">
          {renderInstructions()}
          
          {calibrationStep > 0 && calibrationStep < 5 && (
            <div className="progress-indicator">
              <p>Calibration Progress:</p>
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ width: `${(calibrationStep / 5) * 100}%` }}
                ></div>
              </div>
              <p>{calibrationStep} of 5 steps complete</p>
            </div>
          )}
        </div>
      </div>
      
      {/* Recording Overlay - only shown when actively recording */}
      {isRecording && (
        <div className="recording-overlay">
          <div className="countdown-timer">{countdown}</div>
          {calibrationStep >= 3 ? 
            <div className="recording-message intentional">Recording Intentional Movements</div> : 
            <div className="recording-message unintentional">Recording Unintentional Movements</div>
          }
        </div>
      )}
    </div>
  );
};

export default CalibrationComponent;