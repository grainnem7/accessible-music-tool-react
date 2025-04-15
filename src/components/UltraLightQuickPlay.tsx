import React, { useState, useEffect, useRef } from 'react';
import UltraLightWebcam from './UltraLightWebcam';
import { SoundEngine, SoundPreset } from '../utils/SoundEngine';
import './CalibrationComponent.css';
import './QuickPlay.css';

interface UltraLightQuickPlayProps {
  soundEngine: SoundEngine;
  onBack: () => void;
}

// Enhanced region mappings to make the app more responsive
// Added more regions in the center/face area for better facial detection
const REGION_MAPPINGS = [
  { region: 'top-left', keypoint: 'left_shoulder' },
  { region: 'top-right', keypoint: 'right_shoulder' },
  { region: 'middle-left', keypoint: 'left_wrist' },
  { region: 'middle-right', keypoint: 'right_wrist' },
  { region: 'center', keypoint: 'nose' },
  { region: 'center-top', keypoint: 'left_eye' },  // New mapping for eye region
  { region: 'center-bottom', keypoint: 'right_eye' }, // New mapping for other eye
  { region: 'top-center', keypoint: 'nose' }  // Another mapping for nose area
];

const UltraLightQuickPlay: React.FC<UltraLightQuickPlayProps> = ({ 
  soundEngine, 
  onBack 
}) => {
  // Minimal state to reduce re-renders
  const [selectedPreset, setSelectedPreset] = useState<string>('Piano');
  const [presetDescription, setPresetDescription] = useState<string>("");
  const [isSoundOn, setIsSoundOn] = useState<boolean>(true);
  const [lastKeypoint, setLastKeypoint] = useState<string>("");
  // Add a state for sensitivity
  const [sensitivity, setSensitivity] = useState<number>(30); // Default threshold
  
  // Refs for values that don't need re-renders
  const isSoundOnRef = useRef<boolean>(true);
  const lastTriggerTimeRef = useRef<Record<string, number>>({});
  const cooldownTimeRef = useRef<number>(300); // ms between sound triggers
  const sensitivityRef = useRef<number>(30);

  // Initialize on mount
  useEffect(() => {
    let isMounted = true;
    
    const initApp = async () => {
      try {
        // Initialize audio
        await soundEngine.initialize();
        
        // Load default preset
        soundEngine.loadPreset('Piano');
        
        // Get preset description
        const presetInfo = soundEngine.getPresets().find(p => p.name === 'Piano');
        if (presetInfo && isMounted) {
          setPresetDescription(presetInfo.description);
        }
        
        // Play test sound to confirm audio works
        setTimeout(() => {
          if (isMounted) {
            soundEngine.playTestTone();
          }
        }, 500);
      } catch (error) {
        console.error("Error initializing app:", error);
      }
    };
    
    initApp();
    
    return () => {
      isMounted = false;
      try {
        soundEngine.stopAllSounds();
      } catch (e) {
        console.error("Error cleaning up:", e);
      }
    };
  }, [soundEngine]);
  
  // Keep refs in sync with state
  useEffect(() => {
    isSoundOnRef.current = isSoundOn;
  }, [isSoundOn]);

  useEffect(() => {
    sensitivityRef.current = sensitivity;
  }, [sensitivity]);

  // Handle motion detection with adjustments for mirroring
  const handleMotionDetected = (x: number, y: number, direction: string, velocity: number) => {
    if (!isSoundOnRef.current) return;
    
    // Lower threshold check using the sensitivity state to make it more customizable
    if (velocity < sensitivityRef.current) return;
    
    try {
      // Determine which region of the screen had motion
      const width = 320; // Match webcam dimensions
      const height = 240;
      
      // Simple mapping from screen position to body keypoint
      // The webcam is now mirrored, so the x coordinates are already flipped
      // This means right is right and left is left from the user's perspective
      let keypointName: string;
      
      // Enhanced region detection with more center areas for face detection
      if (y < height / 4) { // Top quarter
        if (x < width / 3) {
          keypointName = 'left_shoulder';
        } else if (x > (width * 2 / 3)) {
          keypointName = 'right_shoulder';
        } else {
          keypointName = 'nose'; // Top center - likely facial movement
        }
      } else if (y < height / 2) { // Upper middle
        if (x < width / 3) {
          keypointName = 'left_elbow';
        } else if (x > (width * 2 / 3)) {
          keypointName = 'right_elbow';
        } else if (x < width / 2) {
          keypointName = 'left_eye'; // Left-center area (facial)
        } else {
          keypointName = 'right_eye'; // Right-center area (facial)
        }
      } else if (y < height * 3 / 4) { // Lower middle
        if (x < width / 3) {
          keypointName = 'left_wrist';
        } else if (x > (width * 2 / 3)) {
          keypointName = 'right_wrist';
        } else {
          keypointName = 'nose'; // Center area - could be nodding
        }
      } else { // Bottom quarter
        keypointName = x < width / 2 ? 'left_hip' : 'right_hip';
      }
      
      // Update UI occasionally (not for every detection)
      if (Math.random() < 0.2) {
        setLastKeypoint(`${keypointName} ${direction}`);
      }
      
      // Check cooldown
      const now = Date.now();
      const lastTrigger = lastTriggerTimeRef.current[keypointName] || 0;
      
      if (now - lastTrigger < cooldownTimeRef.current) {
        return;
      }
      
      // Update last trigger time
      lastTriggerTimeRef.current[keypointName] = now;
      
      // Trigger sound
      soundEngine.processMovement(
        keypointName,
        true,
        direction,
        velocity
      );
    } catch (error) {
      console.error("Error processing motion:", error);
    }
  };

  // UI handlers
  const handlePresetChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    try {
      const presetName = event.target.value;
      console.log(`Changing preset to: ${presetName}`);
      
      // Update UI first for responsiveness
      setSelectedPreset(presetName);
      
      // Find the preset description
      const presets = soundEngine.getPresets();
      const presetInfo = presets.find(p => p.name === presetName);
      
      if (presetInfo) {
        setPresetDescription(presetInfo.description);
      } else {
        setPresetDescription("");
      }
      
      // Apply the preset change in a separate thread
      setTimeout(() => {
        try {
          // First stop any playing sounds
          soundEngine.stopAllSounds();
          
          // Then load the new preset
          soundEngine.loadPreset(presetName);
          
          // Play a test sound after a brief pause
          setTimeout(() => {
            if (isSoundOnRef.current) {
              soundEngine.playTestTone();
            }
          }, 300);
          
          console.log(`Preset changed successfully to: ${presetName}`);
        } catch (e) {
          console.error("Error applying preset:", e);
        }
      }, 100);
    } catch (e) {
      console.error("Error in preset change handler:", e);
    }
  };

  const toggleSound = () => {
    try {
      const newState = !isSoundOn;
      setIsSoundOn(newState);
      isSoundOnRef.current = newState;
      
      if (!newState) {
        setTimeout(() => {
          soundEngine.stopAllSounds();
        }, 50);
      }
    } catch (e) {
      console.error("Error toggling sound:", e);
    }
  };

  const playTestSound = () => {
    try {
      soundEngine.playTestTone();
    } catch (e) {
      console.error("Error playing test sound:", e);
    }
  };

  // Handle sensitivity change
  const handleSensitivityChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value, 10);
    setSensitivity(value);
  };

  // Super minimal UI with added sensitivity control
  return (
    <div className="calibration-container">
      <h1>Quick Play Mode</h1>
      <p className="quick-play-description">
        Move in the webcam view to create music! No calibration needed.
      </p>
      
      <div className="webcam-section" style={{ marginBottom: "20px" }}>
        <UltraLightWebcam 
          onMotionDetected={handleMotionDetected} 
          mirrored={true} // Ensure mirroring is enabled
        />
      </div>
      
      <div className="controls-panel">
        <h2>Music Controls</h2>
        
        <div className="preset-selector">
          <label>Sound Preset:</label>
          <select 
            value={selectedPreset} 
            onChange={handlePresetChange}
          >
            {soundEngine.getPresets().map(preset => (
              <option key={preset.name} value={preset.name}>
                {preset.name}
              </option>
            ))}
          </select>
        </div>
        
        <div className="preset-description">
          {presetDescription}
        </div>
        
        {/* Added sensitivity control slider */}
        <div className="sensitivity-control">
          <label>Motion Sensitivity: {sensitivity}</label>
          <input
            type="range"
            min="5"
            max="50"
            value={sensitivity}
            onChange={handleSensitivityChange}
            className="sensitivity-slider"
          />
          <div className="sensitivity-labels">
            <span>More Sensitive</span>
            <span>Less Sensitive</span>
          </div>
        </div>
        
        {lastKeypoint && (
          <div className="motion-detected">
            Last detected: {lastKeypoint}
          </div>
        )}
        
        <div className="action-buttons">
          <button onClick={toggleSound}>
            {isSoundOn ? 'Mute Sound' : 'Unmute Sound'}
          </button>
          <button onClick={playTestSound}>
            Play Test Sound
          </button>
          <button className="back-button" onClick={onBack}>
            Back to Menu
          </button>
        </div>
        
        <div className="quick-play-instructions">
          <h3>How to Play:</h3>
          <ul>
            <li>Move in the <strong>left area</strong> for left hand sounds</li>
            <li>Move in the <strong>right area</strong> for right hand sounds</li>
            <li>Move your <strong>face, eyes, or nose</strong> for special effects</li>
            <li>Adjust the <strong>sensitivity slider</strong> for better response</li>
            <li>Lower sensitivity (move slider left) to detect smaller movements</li>
          </ul>
          
          <div className={`sound-status ${isSoundOn ? 'on' : 'off'}`}>
            Sound is currently {isSoundOn ? 'ON' : 'OFF'}
          </div>
        </div>
      </div>
    </div>
  );
};

export default UltraLightQuickPlay;