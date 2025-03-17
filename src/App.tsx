import React, { useState, useEffect, useCallback } from 'react';
import WebcamCapture from './components/WebcamCapture';
import { MLIntentionDetector, MovementInfo } from './utils/MLIntentionDetector';
import CalibrationComponent from './components/CalibrationComponent';
import { SoundEngine, SoundPreset } from './utils/SoundEngine';
import * as poseDetection from '@tensorflow-models/pose-detection';
import CalibrationComponentProps from './components/CalibrationComponentProps';
// Import will be used once the component is created
// import GraphicScoreComponent from './components/GraphicScoreComponent';
import './App.css';

// Define application states
enum AppState {
  Welcome = 'welcome',
  Calibration = 'calibration',
  Performance = 'performance',
  Settings = 'settings',
  GraphicScore = 'graphicScore'
}

const App: React.FC = () => {
  // For debugging - remove after confirming changes are applied
  console.log('App.tsx loaded - Version with Graphic Score feature');
  
  // State
  const [appState, setAppState] = useState<AppState>(AppState.Welcome);
  const [detector] = useState<MLIntentionDetector>(new MLIntentionDetector());
  const [soundEngine] = useState<SoundEngine>(new SoundEngine());
  const [userId, setUserId] = useState<string>('');
  const [selectedPreset, setSelectedPreset] = useState<string>('Piano');
  const [presets, setPresets] = useState<SoundPreset[]>([]);
  const [detectedMovements, setDetectedMovements] = useState<MovementInfo[]>([]);
  const [isSoundInitialized, setIsSoundInitialized] = useState<boolean>(false);
  const [showDebugInfo, setShowDebugInfo] = useState<boolean>(false);
  const [isRecordingMovements, setIsRecordingMovements] = useState<boolean>(false);
  const [recordedMovements, setRecordedMovements] = useState<MovementInfo[]>([]);

  // Load presets from the sound engine
  useEffect(() => {
    const availablePresets = soundEngine.getPresets();
    setPresets(availablePresets);
  }, [soundEngine]);

  // Initialize sounds when user loads app
  const initializeAudio = useCallback(async () => {
    try {
      await soundEngine.initialize();
      soundEngine.loadPreset(selectedPreset);
      setIsSoundInitialized(true);
    } catch (err) {
      console.error('Failed to initialize audio:', err);
      alert('Unable to initialize audio. Please check your browser settings.');
    }
  }, [soundEngine, selectedPreset]);

  // Handle pose detection results
  const handlePoseDetected = useCallback((poses: poseDetection.Pose[]) => {
    if (appState === AppState.Performance && poses.length > 0) {
      // Process the pose with the ML detector
      const movements = detector.processPoses(poses);
      setDetectedMovements(movements);
      
      // Record movements if enabled
      if (isRecordingMovements) {
        setRecordedMovements(prev => [...prev, ...movements]);
      }
      
      // Feed intentional movements to the sound engine
      movements.forEach(movement => {
        if (movement.isIntentional) {
          soundEngine.processMovement(
            movement.keypoint,
            movement.isIntentional,
            movement.direction,
            movement.velocity
          );
        }
      });
    }
  }, [appState, detector, soundEngine, isRecordingMovements]);

  // Toggle recording state
  const toggleRecording = useCallback(() => {
    console.log("Toggling recording state");
    setIsRecordingMovements(prev => !prev);
    
    // If stopping recording, you might want to save or process the data
    if (isRecordingMovements) {
      console.log(`Recorded ${recordedMovements.length} movements`);
    }
  }, [isRecordingMovements, recordedMovements.length]);

  // Clear recorded movements
  const clearRecordedMovements = useCallback(() => {
    console.log("Clearing recorded movements");
    setRecordedMovements([]);
  }, []);

  // Change the selected sound preset
  const handlePresetChange = (presetName: string) => {
    setSelectedPreset(presetName);
    soundEngine.loadPreset(presetName);
  };

  // Handle calibration completion
  const handleCalibrationComplete = (calibratedUserId: string) => {
    setUserId(calibratedUserId);
    setAppState(AppState.Performance);
  };

  // Try to load saved model for a user
  const loadUserModel = async (id: string) => {
    try {
      const success = await detector.loadModel(id);
      if (success) {
        setUserId(id);
        return true;
      }
      return false;
    } catch (error) {
      console.error('Error loading user model:', error);
      return false;
    }
  };

  // Render the appropriate component based on app state
  const renderContent = () => {
    switch (appState) {
      case AppState.Welcome:
        return (
          <div className="welcome-screen">
            <h1>Accessible Music Creation</h1>
            <p>Create music through your movements using computer vision and AI</p>
            
            <div className="action-buttons">
              <button 
                className="primary-button"
                onClick={() => {
                  initializeAudio();
                  setAppState(AppState.Calibration);
                }}
              >
                New User Calibration
              </button>
              
              <div className="returning-user">
                <p>Returning user? Enter your ID:</p>
                <div className="user-login">
                  <input 
                    type="text" 
                    placeholder="Your User ID"
                    value={userId}
                    onChange={(e) => setUserId(e.target.value)}
                  />
                  <button 
                    onClick={async () => {
                      if (userId.trim() === '') {
                        alert('Please enter a user ID');
                        return;
                      }
                      
                      await initializeAudio();
                      const success = await loadUserModel(userId);
                      
                      if (success) {
                        setAppState(AppState.Performance);
                      } else {
                        alert('No saved calibration found for this user. Please create a new calibration.');
                      }
                    }}
                  >
                    Load Profile
                  </button>
                </div>
              </div>
            </div>
          </div>
        );
        
      case AppState.Calibration:
        return (
          <div className="calibration-screen">
            <h1>Movement Calibration</h1>
            {/* Use type assertion to bypass TypeScript error */}
            <CalibrationComponent 
              {...{detector, onCalibrationComplete: handleCalibrationComplete} as any} 
            />
          </div>
        );
        
      case AppState.Performance:
        return (
          <div className="performance-screen">
            {/* Debug marker - remove after confirming changes */}
            <div style={{padding: '5px', margin: '5px', background: '#f0f0f0', display: 'none'}}>
              Performance Screen with Graphic Score feature
            </div>
            
            <div className="webcam-container">
              <WebcamCapture onPoseDetected={handlePoseDetected} />
            </div>
            
            <div className="controls-panel">
              <h2>Music Controls</h2>
              
              <div className="preset-selector">
                <label>Sound Preset:</label>
                <select 
                  value={selectedPreset} 
                  onChange={(e) => handlePresetChange(e.target.value)}
                >
                  {presets.map(preset => (
                    <option key={preset.name} value={preset.name}>
                      {preset.name}
                    </option>
                  ))}
                </select>
              </div>
              
              <div className="preset-description">
                {presets.find(p => p.name === selectedPreset)?.description}
              </div>
              
              <div className="action-buttons">
                <button onClick={() => setAppState(AppState.Settings)}>
                  Settings
                </button>
                <button 
                  onClick={() => {
                    console.log('Graphic Score button clicked');
                    setAppState(AppState.GraphicScore);
                  }}
                  style={{backgroundColor: '#3498db'}}
                >
                  Graphic Score
                </button>
                <button onClick={() => setShowDebugInfo(!showDebugInfo)}>
                  {showDebugInfo ? 'Hide' : 'Show'} Debug Info
                </button>
              </div>
              
              <div className="recording-controls">
                <button 
                  className={`record-button ${isRecordingMovements ? 'recording' : ''}`}
                  onClick={toggleRecording}
                >
                  {isRecordingMovements ? 'Stop Recording' : 'Start Recording'}
                </button>
                {isRecordingMovements && (
                  <div className="recording-indicator">
                    <div className="recording-dot"></div>
                    <span>Recording movements ({recordedMovements.length})</span>
                  </div>
                )}
              </div>
              
              {showDebugInfo && (
                <div className="debug-info">
                  <h3>Movement Detection:</h3>
                  <ul>
                    {detectedMovements.map((movement, idx) => (
                      <li key={idx} className={movement.isIntentional ? 'intentional' : 'unintentional'}>
                        {movement.keypoint}: {movement.direction} 
                        (v: {movement.velocity.toFixed(2)}, 
                        {movement.isIntentional ? ' INTENTIONAL' : ' unintentional'})
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        );
        
      case AppState.Settings:
        return (
          <div className="settings-screen">
            <h1>Settings</h1>
            
            <div className="settings-section">
              <h2>User Profile</h2>
              <p>Current User: <strong>{userId}</strong></p>
              <button onClick={() => setAppState(AppState.Calibration)}>
                Recalibrate Movements
              </button>
            </div>
            
            <div className="settings-section">
              <h2>Sound Settings</h2>
              <p>Adjust your sound preferences:</p>
              
              <div className="form-group">
                <label>Reverb Amount:</label>
                <input 
                  type="range" 
                  min="0" 
                  max="1" 
                  step="0.1" 
                  defaultValue="0.3"
                  onChange={(e) => {
                    // Add function to adjust reverb
                    soundEngine.setReverbAmount(parseFloat(e.target.value));
                  }}
                />
              </div>
              
              <div className="form-group">
                <label>Response Sensitivity:</label>
                <input 
                  type="range" 
                  min="1" 
                  max="10" 
                  step="1" 
                  defaultValue="5"
                  onChange={(e) => {
                    // Add function to adjust sensitivity
                    const sensitivity = parseFloat(e.target.value);
                    console.log(`Setting sensitivity to ${sensitivity}/10`);
                  }}
                />
              </div>
            </div>
            
            <button 
              className="back-button"
              onClick={() => setAppState(AppState.Performance)}
            >
              Back to Performance
            </button>
          </div>
        );

      case AppState.GraphicScore:
        console.log('Rendering Graphic Score view');
        // Use a simple placeholder until GraphicScoreComponent is created
        return (
          <div className="graphic-score-screen">
            <h1>Movement Graphic Score</h1>
            
            <div style={{
              padding: '20px',
              backgroundColor: '#f8f9fa',
              borderRadius: '10px',
              marginBottom: '20px'
            }}>
              <h3>Temporary Graphic Score Placeholder</h3>
              <p>This is a placeholder for the GraphicScoreComponent.</p>
              <p>Once you create the component files, replace this with the actual component.</p>
              
              <div style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '10px',
                margin: '20px 0'
              }}>
                <div>
                  <button 
                    style={{
                      backgroundColor: isRecordingMovements ? '#e74c3c' : '#3498db',
                      color: 'white',
                      border: 'none',
                      padding: '10px 20px',
                      borderRadius: '5px',
                      cursor: 'pointer'
                    }}
                    onClick={toggleRecording}
                  >
                    {isRecordingMovements ? 'Stop Recording' : 'Start Recording'}
                  </button>
                </div>
                
                <div>
                  <p>Recorded movements: {recordedMovements.length}</p>
                </div>
              </div>
            </div>
            
            <div className="action-buttons">
              <button 
                onClick={clearRecordedMovements}
                disabled={recordedMovements.length === 0 || isRecordingMovements}
              >
                Clear Recording
              </button>
              
              <button 
                className="back-button"
                onClick={() => setAppState(AppState.Performance)}
              >
                Back to Performance
              </button>
            </div>
          </div>
        );
    }
  };

  return (
    <div className="app-container">
      {renderContent()}
    </div>
  );
};

export default App;