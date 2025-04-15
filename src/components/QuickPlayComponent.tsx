import React, { useState, useEffect, useCallback } from 'react';
import WebcamCapture from './WebcamCapture';
import { SoundEngine, SoundPreset, SoundMapping } from '../utils/SoundEngine';
import * as poseDetection from '@tensorflow-models/pose-detection';
import './CalibrationComponent.css'; // Reuse calibration styles
import './QuickPlay.css'; // Add our Quick Play specific styles

interface QuickPlayComponentProps {
  soundEngine: SoundEngine;
  onBack: () => void;
}

// Common keypoints to track for movement
const TRACKED_KEYPOINTS = [
  'left_wrist', 'right_wrist',
  'left_elbow', 'right_elbow',
  'left_shoulder', 'right_shoulder',
  'nose'
];

// Velocity threshold for triggering sounds
const VELOCITY_THRESHOLD = 10;

// Cooldown between triggers (ms)
const COOLDOWN_PERIOD = 300;

const QuickPlayComponent: React.FC<QuickPlayComponentProps> = ({ 
  soundEngine, 
  onBack 
}) => {
  const [presets, setPresets] = useState<SoundPreset[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<string>('Piano');
  const [detectedMovements, setDetectedMovements] = useState<{keypoint: string, direction: string, velocity: number, x: number, y: number}[]>([]);
  const [showDebugInfo, setShowDebugInfo] = useState<boolean>(false);
  const [lastTriggerTime, setLastTriggerTime] = useState<Record<string, number>>({});
  const [isSoundOn, setIsSoundOn] = useState<boolean>(true);
  const [prevPositions, setPrevPositions] = useState<Record<string, {x: number, y: number}>>({});
  const [isAudioInitialized, setIsAudioInitialized] = useState<boolean>(false);

  // Initialize audio and load presets
  useEffect(() => {
    const initializeAudio = async () => {
      try {
        // Make sure audio is initialized
        await soundEngine.initialize();
        setIsAudioInitialized(true);
        console.log("QuickPlay: Audio initialized successfully");
        
        // Load presets
        const availablePresets = soundEngine.getPresets();
        setPresets(availablePresets);
        
        // Load default preset
        soundEngine.loadPreset(selectedPreset);
      } catch (err) {
        console.error('QuickPlay: Failed to initialize audio:', err);
        alert('Unable to initialize audio. Please check your browser settings and make sure you allowed microphone access.');
      }
    };

    initializeAudio();
    
    // Cleanup function
    return () => {
      soundEngine.stopAllSounds();
    };
  }, [soundEngine, selectedPreset]);

  // Handle pose detection results
  const handlePoseDetected = useCallback((poses: poseDetection.Pose[]) => {
    if (poses.length === 0 || !isAudioInitialized) return;
    
    const movements: {keypoint: string, direction: string, velocity: number, x: number, y: number}[] = [];
    const now = Date.now();
    const newPositions: Record<string, {x: number, y: number}> = {};
    
    // Only process the first detected person
    const pose = poses[0];
    
    // Check each tracked keypoint
    TRACKED_KEYPOINTS.forEach(keypointName => {
      const keypoint = pose.keypoints.find(kp => kp.name === keypointName);
      
      if (keypoint && keypoint.score && keypoint.score > 0.3) { // Lower threshold for detection
        newPositions[keypointName] = { x: keypoint.x, y: keypoint.y };
        
        // Calculate velocity if we have previous position
        const prevPosition = prevPositions[keypointName];
        if (prevPosition) {
          const dx = keypoint.x - prevPosition.x;
          const dy = keypoint.y - prevPosition.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          
          // Determine direction of movement
          let direction = 'none';
          if (Math.abs(dx) > Math.abs(dy)) {
            direction = dx > 0 ? 'right' : 'left';
          } else {
            direction = dy > 0 ? 'down' : 'up';
          }
          
          // Lower threshold to detect more movements
          if (distance > 5) {
            // Relax cooldown to allow more frequent sounds
            const lastTrigger = lastTriggerTime[keypointName] || 0;
            if (now - lastTrigger > 200) {
              movements.push({
                keypoint: keypointName,
                direction,
                velocity: distance,
                x: keypoint.x,
                y: keypoint.y
              });
              
              // Update last trigger time
              setLastTriggerTime(prev => ({
                ...prev,
                [keypointName]: now
              }));
              
              // Process the movement in sound engine
              if (isSoundOn) {
                console.log(`Playing sound for ${keypointName} ${direction} with velocity ${distance}`);
                
                // Always play a sound regardless of mapping
                // This ensures something happens when movement is detected
                if (keypointName.includes('wrist')) {
                  soundEngine.processMovement(
                    keypointName,
                    true, 
                    direction,
                    distance
                  );
                } else if (keypointName.includes('shoulder')) {
                  soundEngine.processMovement(
                    keypointName,
                    true, 
                    'up', // Use fixed direction for shoulders
                    distance
                  );
                } else if (keypointName === 'nose') {
                  soundEngine.processMovement(
                    'nose',
                    true, 
                    'right', // Use fixed direction for nose
                    distance
                  );
                }
              }
            }
          }
        }
      }
    });
    
    // Update previous positions for next frame
    setPrevPositions(newPositions);
    
    // Only update movements if we have some
    if (movements.length > 0) {
      setDetectedMovements(movements);
    }
  }, [prevPositions, lastTriggerTime, soundEngine, isSoundOn, isAudioInitialized]);

  // Change the selected sound preset
  const handlePresetChange = (presetName: string) => {
    setSelectedPreset(presetName);
    soundEngine.loadPreset(presetName);
    console.log(`Preset changed to: ${presetName}`);
  };

  // Toggle sound on/off
  const toggleSound = () => {
    if (isSoundOn) {
      soundEngine.stopAllSounds();
    }
    setIsSoundOn(!isSoundOn);
    console.log(`Sound is now ${!isSoundOn ? 'ON' : 'OFF'}`);
  };

  // Play a test sound to verify audio is working
  const playTestSound = () => {
    console.log("Playing test sound...");
    if (!isAudioInitialized) {
      soundEngine.initialize().then(() => {
        setIsAudioInitialized(true);
        soundEngine.playTestTone();
      });
    } else {
      soundEngine.playTestTone();
    }
  };

  return (
    <div className="calibration-container">
      <h1>Quick Play Mode</h1>
      <p className="quick-play-description">
        Move your body to create music!
      </p>
      
      <div className="calibration-layout">
        <div className="webcam-section">
          <WebcamCapture 
            onPoseDetected={handlePoseDetected} 
            showDebugInfo={showDebugInfo}
          />
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
            <button onClick={toggleSound}>
              {isSoundOn ? 'Mute Sound' : 'Unmute Sound'}
            </button>
            <button onClick={playTestSound}>
              Play Test Sound
            </button>
            <button onClick={() => setShowDebugInfo(!showDebugInfo)}>
              {showDebugInfo ? 'Hide' : 'Show'} Debug Info
            </button>
            <button className="back-button" onClick={onBack}>
              Back to Menu
            </button>
          </div>
          
          {showDebugInfo && (
            <div className="debug-info">
              <h3>Movement Detection:</h3>
              <ul>
                {detectedMovements.map((movement, idx) => (
                  <li key={idx} className="intentional">
                    {movement.keypoint}: {movement.direction} 
                    (velocity: {movement.velocity.toFixed(2)})
                  </li>
                ))}
              </ul>
              <p>Audio initialized: {isAudioInitialized ? 'Yes' : 'No'}</p>
              <p>Sound is: {isSoundOn ? 'ON' : 'OFF'}</p>
              <p>Active preset: {selectedPreset}</p>
            </div>
          )}
          
          <div className="quick-play-instructions">
            <h3>How to Play:</h3>
            <ul>
              <li>Move your <strong>wrists</strong> up/down for melodies</li>
              <li>Try <strong>shoulder</strong> movements for different sounds</li>
              <li>Move your head (<strong>nose</strong>) for special effects</li>
              <li>Change presets to explore different instruments</li>
            </ul>
            
            <div className={`sound-status ${isSoundOn ? 'on' : 'off'}`}>
              Sound is currently {isSoundOn ? 'ON' : 'OFF'}
            </div>
            
            <h3>Current Preset Gestures:</h3>
            <div className="gesture-visualization">
              {presets.find(p => p.name === selectedPreset)?.mappings.map((mapping, index) => (
                <div key={index} className="gesture-card">
                  <div className="gesture-name">{mapping.keypoint.replace('_', ' ')}</div>
                  <div className="gesture-direction">Direction: {mapping.direction}</div>
                  <div className="gesture-sound">
                    Sound: {mapping.soundType === 'chord' ? 'Chord' : 
                           mapping.soundType === 'note' ? 'Note' : 
                           mapping.soundType === 'drum' ? 'Drum' : 'Effect'}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default QuickPlayComponent;