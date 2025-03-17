import React, { useState, useEffect } from 'react';
import { MLIntentionDetector } from '../utils/MLIntentionDetector';
import WebcamCapture from './WebcamCapture';
import * as poseDetection from '@tensorflow-models/pose-detection';
import './CalibrationComponent.css';
import { azureConfig } from '../config';
import AzureIntegrationSettings from './AzureIntegrationSettings';

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
    isTfInitialized: false,
    calibrationQuality: 0,
    azureEnabled: false
  });
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainingProgress, setTrainingProgress] = useState<number>(0);
  const [showDebugInfo, setShowDebugInfo] = useState<boolean>(true); 
  const [calibrationDuration, setCalibrationDuration] = useState<number>(15);
  const [lastDetectedMovements, setLastDetectedMovements] = useState<string[]>([]);
  const [qualityFeedback, setQualityFeedback] = useState<string>('');
  const [useAzure, setUseAzure] = useState<boolean>(true);
  const [isAzureTraining, setIsAzureTraining] = useState<boolean>(false);
  const [azureProgress, setAzureProgress] = useState<number>(0);
  const [intentionalKeypoints, setIntentionalKeypoints] = useState<string[]>([]);
  const [advancedOptions, setAdvancedOptions] = useState<boolean>(false);
  const [showAzureSettings, setShowAzureSettings] = useState<boolean>(false);

  // Initialize calibration duration from detector
  useEffect(() => {
    if (detector) {
      setCalibrationDuration(detector.getCalibrationDuration());
      
      // Initialize with Azure configuration
      if (useAzure) {
        detector.configureAzureServices({
          apiKey: azureConfig.computerVisionKey,
          endpoint: azureConfig.computerVisionEndpoint,
          enabled: true
        });
        
        setModelStatus(prev => ({ ...prev, azureEnabled: true }));
      }
    }
  }, [detector]);
  
  // Timer for collecting samples
  useEffect(() => {
    let timer: NodeJS.Timeout;
    
    if (countdown > 0) {
      timer = setTimeout(() => setCountdown(countdown - 1), 1000);
    } else if (countdown === 0 && isRecording) {
      setIsRecording(false);
      
      // Show quality feedback based on calibration data
      updateQualityFeedback();
      
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
  
  // Update calibration quality feedback
  const updateQualityFeedback = () => {
    const status = detector.getStatus();
    
    if (status.calibrationSamples < 10) {
      setQualityFeedback('Need more calibration samples. Please continue.');
      return;
    }
    
    // Check balance between intentional and unintentional samples
    const intentionalRatio = status.intentionalSamples / status.calibrationSamples;
    
    if (intentionalRatio < 0.3 || intentionalRatio > 0.7) {
      setQualityFeedback('Your calibration is imbalanced. Try to provide both intentional and unintentional movements.');
      return;
    }
    
    // Quality based on calibration quality score
    if (status.calibrationQuality < 40) {
      setQualityFeedback('Low calibration quality. Try making more varied and distinct movements.');
    } else if (status.calibrationQuality < 70) {
      setQualityFeedback('Moderate calibration quality. Additional samples may improve accuracy.');
    } else {
      setQualityFeedback('Good calibration quality. Ready for training!');
    }
  };
  
  // Process poses for calibration
  const handlePoseDetected = (poses: poseDetection.Pose[]) => {
    if (!poses || poses.length === 0) return;
    
    if (isRecording) {
      // During calibration steps 1 & 2, we're collecting unintentional movements
      // During steps 3 & 4, we're collecting intentional movements
      const isIntentional = calibrationStep >= 3 && calibrationStep <= 4;
      detector.addCalibrationSample(isIntentional);
      
      // For visualization, track which keypoints were detected with high confidence
      const keypoints = poses[0].keypoints
        .filter(kp => kp.score && kp.score > 0.5 && kp.name)
        .map(kp => kp.name || '');
      
      setLastDetectedMovements(keypoints);
      
      // Store intentional keypoints for highlighting during training
      if (isIntentional && keypoints.length > 0) {
        setIntentionalKeypoints(prev => {
          const updated = [...prev];
          keypoints.forEach(name => {
            if (!updated.includes(name)) {
              updated.push(name);
            }
          });
          return updated;
        });
      }
    }
  };

  // Start recording samples for the current step
  const startRecording = () => {
    setIsRecording(true);
    setCountdown(calibrationDuration); // Use configured duration
    setLastDetectedMovements([]);
  };
  
  // Train the model with collected samples
  const trainModel = async () => {
    // Configure Azure if enabled
    if (useAzure) {
      detector.configureAzureServices({
        apiKey: azureConfig.computerVisionKey,
        endpoint: azureConfig.computerVisionEndpoint,
        enabled: true
      });
      
      setModelStatus(prev => ({ ...prev, azureEnabled: true }));
    }
    
    setIsTraining(true);
    setTrainingProgress(0);
    
    // Set up progress tracking
    detector.setTrainingProgressCallback((progress: number) => {
      setTrainingProgress(Math.round(progress * 100));
    });
    
    const success = await detector.trainModel();
    
    // If Azure is enabled, also train Azure model
    if (useAzure) {
      setIsAzureTraining(true);
      
      try {
        // Simulate Azure training progress
        const azureTrainingInterval = setInterval(() => {
          setAzureProgress(prev => {
            const nextProgress = prev + Math.random() * 5;
            return nextProgress >= 100 ? 100 : nextProgress;
          });
        }, 300);
        
        // Train Azure model
        const azureSuccess = await detector.trainAzureModel(userId);
        
        clearInterval(azureTrainingInterval);
        setAzureProgress(100);
        
        if (!azureSuccess) {
          alert("Azure training encountered issues. Local model is still available.");
        }
      } catch (error) {
        console.error("Azure training error:", error);
        alert("Azure training failed. Using local model only.");
      } finally {
        setIsAzureTraining(false);
      }
    }
    
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
    setQualityFeedback('');
    setLastDetectedMovements([]);
    setIntentionalKeypoints([]);
  };
  
  // Handle calibration duration change
  const handleDurationChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newDuration = parseInt(e.target.value, 10);
    setCalibrationDuration(newDuration);
    detector.setCalibrationDuration(newDuration);
  };
  
  // Toggle advanced options
  const toggleAdvancedOptions = () => {
    setAdvancedOptions(!advancedOptions);
  };
  
  // Toggle Azure settings panel
  const toggleAzureSettings = () => {
    setShowAzureSettings(!showAzureSettings);
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
            
            <div className="settings-panel">
              <div className="settings-header" onClick={toggleAdvancedOptions}>
                <h3>Calibration Settings {advancedOptions ? '▼' : '▶'}</h3>
              </div>
              
              {advancedOptions && (
                <div className="advanced-settings">
                  <div className="setting-item">
                    <label>Sample Duration:</label>
                    <select 
                      value={calibrationDuration}
                      onChange={handleDurationChange}
                      className="duration-select"
                    >
                      <option value="5">5 seconds</option>
                      <option value="10">10 seconds</option>
                      <option value="15">15 seconds (recommended)</option>
                      <option value="20">20 seconds</option>
                      <option value="30">30 seconds (best quality)</option>
                    </select>
                  </div>
                  
                  <div className="setting-item">
                    <label>Use Azure AI Services:</label>
                    <div className="toggle-container">
                      <input 
                        type="checkbox" 
                        checked={useAzure}
                        onChange={(e) => setUseAzure(e.target.checked)}
                        id="azure-toggle"
                      />
                      <label htmlFor="azure-toggle" className="toggle-label"></label>
                    </div>
                    <div className="setting-description">
                      Using Azure Computer Vision for enhanced gesture recognition
                    </div>
                    
                    {useAzure && (
                      <button 
                        onClick={toggleAzureSettings}
                        className="azure-settings-button"
                      >
                        {showAzureSettings ? 'Hide' : 'Show'} Azure Settings
                      </button>
                    )}
                  </div>
                  
                  {showAzureSettings && useAzure && (
                    <div className="azure-settings-panel">
                      <AzureIntegrationSettings detector={detector} userId={userId} />
                    </div>
                  )}
                </div>
              )}
            </div>
            
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
            
            {qualityFeedback && (
              <div className={`quality-feedback ${modelStatus.calibrationQuality > 70 ? 'good' : 
                                                modelStatus.calibrationQuality > 40 ? 'medium' : 'poor'}`}>
                {qualityFeedback}
              </div>
            )}
            
            {isRecording ? (
              <div className="recording-indicator">
                <div className="recording-dot"></div>
                <p>Recording... {countdown} seconds remaining</p>
                <div className="detected-keypoints">
                  Detected: {lastDetectedMovements.join(', ')}
                </div>
              </div>
            ) : (
              <button onClick={startRecording} className="record-button">
                Start Recording ({calibrationDuration}s)
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
            
            {qualityFeedback && (
              <div className={`quality-feedback ${modelStatus.calibrationQuality > 70 ? 'good' : 
                                                modelStatus.calibrationQuality > 40 ? 'medium' : 'poor'}`}>
                {qualityFeedback}
              </div>
            )}
            
            {isRecording ? (
              <div className="recording-indicator">
                <div className="recording-dot"></div>
                <p>Recording... {countdown} seconds remaining</p>
                <div className="detected-keypoints">
                  Detected: {lastDetectedMovements.join(', ')}
                </div>
              </div>
            ) : (
              <button onClick={startRecording} className="record-button">
                Start Recording ({calibrationDuration}s)
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
            
            {qualityFeedback && (
              <div className={`quality-feedback ${modelStatus.calibrationQuality > 70 ? 'good' : 
                                                modelStatus.calibrationQuality > 40 ? 'medium' : 'poor'}`}>
                {qualityFeedback}
              </div>
            )}
            
            {isRecording ? (
              <div className="recording-indicator intentional">
                <div className="recording-dot"></div>
                <p>Recording... {countdown} seconds remaining</p>
                <div className="detected-keypoints">
                  Detected: {lastDetectedMovements.join(', ')}
                </div>
              </div>
            ) : (
              <button onClick={startRecording} className="record-button intentional">
                Start Recording Intentional Movements ({calibrationDuration}s)
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
            
            {qualityFeedback && (
              <div className={`quality-feedback ${modelStatus.calibrationQuality > 70 ? 'good' : 
                                                modelStatus.calibrationQuality > 40 ? 'medium' : 'poor'}`}>
                {qualityFeedback}
              </div>
            )}
            
            {isRecording ? (
              <div className="recording-indicator intentional">
                <div className="recording-dot"></div>
                <p>Recording... {countdown} seconds remaining</p>
                <div className="detected-keypoints">
                  Detected: {lastDetectedMovements.join(', ')}
                </div>
              </div>
            ) : (
              <button onClick={startRecording} className="record-button intentional">
                Start Recording Intentional Movements ({calibrationDuration}s)
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
            
            {qualityFeedback && (
              <div className={`quality-feedback ${modelStatus.calibrationQuality > 70 ? 'good' : 
                                               modelStatus.calibrationQuality > 40 ? 'medium' : 'poor'}`}>
                <strong>Calibration Quality: {modelStatus.calibrationQuality}/100</strong><br/>
                {qualityFeedback}
              </div>
            )}
            
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
                
                {isAzureTraining && (
                  <div className="azure-training">
                    <p>Training Azure AI model...</p>
                    <div className="progress-container">
                      <div 
                        className="progress-bar-azure" 
                        style={{width: `${azureProgress}%`}}
                      ></div>
                    </div>
                    <p>{Math.round(azureProgress)}% complete</p>
                  </div>
                )}
                
                <div className="loading-spinner"></div>
              </div>
            ) : (
              <>
                <button 
                  onClick={trainModel} 
                  className="train-button" 
                  disabled={modelStatus.calibrationSamples < 10}
                >
                  {useAzure ? "Train Model with Azure AI" : "Train Model"}
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
            
            {useAzure && (
              <div className="azure-complete">
                <div className="azure-badge">
                  <span className="azure-icon">✓</span> Azure AI Enhanced
                </div>
                <p>Your model is enhanced with Azure Computer Vision for better accuracy.</p>
              </div>
            )}
            
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
      <div className="calibration-layout">
        <div className="webcam-section">
          <WebcamCapture 
            onPoseDetected={handlePoseDetected} 
            showDebugInfo={showDebugInfo}
            highlightIntentional={calibrationStep >= 3}
            intentionalKeypoints={intentionalKeypoints}
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
            
            {modelStatus.calibrationQuality > 0 && (
              <div className="sample-counter">
                <div className="sample-label quality">Quality:</div>
                <div className={`sample-value quality-${modelStatus.calibrationQuality > 70 ? 'good' : 
                                                       modelStatus.calibrationQuality > 40 ? 'medium' : 'poor'}`}>
                  {modelStatus.calibrationQuality}/100
                </div>
              </div>
            )}
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
          {lastDetectedMovements.length > 0 && (
            <div className="overlay-keypoints">
              Detected keypoints: {lastDetectedMovements.join(', ')}
            </div>
          )}
        </div>
      )}
      
      {/* Debug toggle */}
      <button 
        onClick={() => setShowDebugInfo(!showDebugInfo)} 
        className="debug-toggle"
      >
        {showDebugInfo ? 'Hide' : 'Show'} Debug Info
      </button>
    </div>
  );
};

export default CalibrationComponent;