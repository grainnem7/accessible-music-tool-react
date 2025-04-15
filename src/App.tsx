import React, { useState, useEffect, useCallback } from 'react';
import WebcamCapture from './components/WebcamCapture';
import { MLIntentionDetector, MovementInfo } from './utils/MLIntentionDetector';
import CalibrationComponent from './components/CalibrationComponent';
import UltraLightQuickPlay from './components/UltraLightQuickPlay';
import { SoundEngine, SoundPreset } from './utils/SoundEngine';
import * as poseDetection from '@tensorflow-models/pose-detection';
import './App.css';

// Define application states
enum AppState {
  Welcome = 'welcome',
  Calibration = 'calibration',
  Performance = 'performance',
  Settings = 'settings',
  QuickPlay = 'quickplay'
}

const App: React.FC = () => {
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
  }, [appState, detector, soundEngine]);

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
                onClick={async () => {
                  try {
                    await initializeAudio();
                    setAppState(AppState.QuickPlay);
                  } catch (err) {
                    console.error("Error starting Quick Play mode:", err);
                  }
                }}
              >
                Quick Play Mode
              </button>
              
              <button 
                className="primary-button"
                onClick={() => {
                  initializeAudio();
                  setAppState(AppState.Calibration);
                }}
              >
                Calibration Mode
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
            <CalibrationComponent 
              detector={detector}
              onCalibrationComplete={handleCalibrationComplete}
            />
          </div>
        );
        
      case AppState.Performance:
        return (
          <div className="performance-screen">
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
                <button onClick={() => setShowDebugInfo(!showDebugInfo)}>
                  {showDebugInfo ? 'Hide' : 'Show'} Debug Info
                </button>
                <button onClick={() => setAppState(AppState.Welcome)}>
                  Back to Menu
                </button>
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
        
      case AppState.QuickPlay:
        return (
          <UltraLightQuickPlay 
            soundEngine={soundEngine}
            onBack={() => setAppState(AppState.Welcome)} 
          />
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
                    const value = parseFloat(e.target.value);
                    soundEngine.setReverbAmount(value);
                  }}
                />
              </div>
              
              <div className="form-group">
                <label>Volume:</label>
                <input 
                  type="range" 
                  min="0" 
                  max="1" 
                  step="0.1" 
                  defaultValue="0.7"
                  onChange={(e) => {
                    const value = parseFloat(e.target.value);
                    soundEngine.setVolume(value);
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
    }
  };

  return (
    <div className="app-container">
      {renderContent()}
    </div>
  );
};

export default App;