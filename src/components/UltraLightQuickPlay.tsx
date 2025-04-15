import React, { useState, useEffect, useRef } from 'react';
import UltraLightWebcam from './UltraLightWebcam';
import { SoundEngine, SoundPreset } from '../utils/SoundEngine';
import './CalibrationComponent.css';
import './QuickPlay.css';

interface UltraLightQuickPlayProps {
  soundEngine: SoundEngine;
  onBack: () => void;
}

// Common mappings from screen regions to keypoints 
// This is a much simpler approach than full pose detection
const REGION_MAPPINGS = [
  { region: 'top-left', keypoint: 'left_shoulder' },
  { region: 'top-right', keypoint: 'right_shoulder' },
  { region: 'middle-left', keypoint: 'left_wrist' },
  { region: 'middle-right', keypoint: 'right_wrist' },
  { region: 'center', keypoint: 'nose' }
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
  
  // Refs for values that don't need re-renders
  const isSoundOnRef = useRef<boolean>(true);
  const lastTriggerTimeRef = useRef<Record<string, number>>({});
  const cooldownTimeRef = useRef<number>(300); // ms between sound triggers

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
  
  // Keep ref in sync with state
  useEffect(() => {
    isSoundOnRef.current = isSoundOn;
  }, [isSoundOn]);

  // Handle motion detection
  const handleMotionDetected = (x: number, y: number, direction: string, velocity: number) => {
    if (!isSoundOnRef.current) return;
    
    try {
      // Determine which region of the screen had motion
      const width = 320; // Match webcam dimensions
      const height = 240;
      
      // Simple mapping from screen position to body keypoint
      let keypointName: string;
      
      if (y < height / 3) {
        // Top third
        keypointName = x < width / 2 ? 'left_shoulder' : 'right_shoulder';
      } else if (y < height * 2 / 3) {
        // Middle third
        keypointName = x < width / 2 ? 'left_wrist' : 'right_wrist';
      } else {
        // Bottom third (or use nose for center regardless of position)
        keypointName = 'nose';
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

  // Super minimal UI
  return (
    <div className="calibration-container">
      <h1>Quick Play Mode</h1>
      <p className="quick-play-description">
        Move in the webcam view to create music! No calibration needed.
      </p>
      
      <div className="webcam-section" style={{ marginBottom: "20px" }}>
        <UltraLightWebcam onMotionDetected={handleMotionDetected} />
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
            <li>Move in the <strong>center</strong> for special effects</li>
            <li>Change presets to explore different instruments</li>
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