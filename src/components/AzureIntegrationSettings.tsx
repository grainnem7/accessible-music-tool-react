// src/components/AzureIntegrationSettings.tsx

import React, { useState, useEffect } from 'react';
import { MLIntentionDetector } from '../utils/MLIntentionDetector';
import { AzureSpeechService, SpeechCommand } from '../services/azureSpeechService';
import { AzureVisionService } from '../services/AzureVisionService';
import { AzureCustomVisionService } from '../services/azureCustomVisionService';
import { AzureMLService } from '../services/azureMLService';
import './AzureIntegrationSettings.css'; // Import external CSS file instead

interface AzureIntegrationSettingsProps {
  detector: MLIntentionDetector;
  userId: string;
}

const AzureIntegrationSettings: React.FC<AzureIntegrationSettingsProps> = ({ 
  detector, 
  userId 
}) => {
  // Azure service states
  const [useAzureVision, setUseAzureVision] = useState<boolean>(false);
  const [useAzureCustomVision, setUseAzureCustomVision] = useState<boolean>(false);
  const [useAzureSpeech, setUseAzureSpeech] = useState<boolean>(false);
  const [useAzureML, setUseAzureML] = useState<boolean>(false);
  
  // Service instances
  const [speechService] = useState<AzureSpeechService>(new AzureSpeechService());
  const [visionService] = useState<AzureVisionService>(new AzureVisionService());
  const [customVisionService] = useState<AzureCustomVisionService>(new AzureCustomVisionService());
  const [mlService] = useState<AzureMLService>(new AzureMLService());
  
  // Status state
  const [trainingStatus, setTrainingStatus] = useState<string>('');
  const [azureTestResponse, setAzureTestResponse] = useState<string>('');

  // Load saved settings
  useEffect(() => {
    if (userId) {
      // Load settings from local storage
      const hasAzureCustomVision = localStorage.getItem(`user-azure-model-${userId}`) === 'true';
      const hasAzureML = localStorage.getItem(`user-azure-ml-model-${userId}`) === 'true';
      
      setUseAzureCustomVision(hasAzureCustomVision);
      setUseAzureML(hasAzureML);
      
      // Update detector settings if needed
      if (detector && hasAzureCustomVision) {
        detector.setUseAzureCustomVision?.(true);
      }
      
      if (detector && hasAzureML) {
        detector.setUseAzureML?.(true);
      }
    }
  }, [userId, detector]);
  
  // Cleanup speech service when component unmounts
  useEffect(() => {
    return () => {
      if (useAzureSpeech) {
        speechService.stopListening();
      }
    };
  }, [useAzureSpeech, speechService]);
  
  // Handle speech command
  const handleSpeechCommand = (command: SpeechCommand) => {
    setAzureTestResponse(`Received voice command: ${command.command} at ${new Date(command.timestamp).toLocaleTimeString()}`);
    // Add your command handling logic here
  };
  
  // Toggle speech recognition
  const toggleSpeechRecognition = () => {
    if (useAzureSpeech) {
      speechService.stopListening();
      setUseAzureSpeech(false);
    } else {
      speechService.startListening(handleSpeechCommand);
      setUseAzureSpeech(true);
    }
  };
  
  // Test Azure Vision Service
  const testAzureVision = async () => {
    try {
      setAzureTestResponse('Testing Azure Computer Vision...');
      
      // You'll need to implement this method to get a sample image for testing
      const testImage = await getSampleImageBlob();
      
      const result = await visionService.detectPose(testImage);
      
      if (result && result.keypoints.length > 0) {
        setAzureTestResponse(`Azure Vision successful! Detected ${result.keypoints.length} keypoints.`);
      } else {
        setAzureTestResponse('Azure Vision test failed. No keypoints detected.');
      }
    } catch (error) {
      console.error('Error testing Azure Vision:', error);
      setAzureTestResponse(`Azure Vision error: ${String(error)}`);
    }
  };
  
  // Helper function to get a sample image for testing
  const getSampleImageBlob = async (): Promise<Blob> => {
    // In a real implementation, you might:
    // 1. Capture from webcam
    // 2. Use a pre-loaded test image
    // 3. Get from user upload
    
    // For this example, we'll create a canvas and draw a simple shape
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Could not get canvas context');
    
    // Draw a figure representing a person
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Head
    ctx.beginPath();
    ctx.arc(320, 100, 50, 0, Math.PI * 2);
    ctx.fillStyle = 'white';
    ctx.fill();
    
    // Body
    ctx.beginPath();
    ctx.moveTo(320, 150);
    ctx.lineTo(320, 300);
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 5;
    ctx.stroke();
    
    // Arms
    ctx.beginPath();
    ctx.moveTo(320, 200);
    ctx.lineTo(250, 250);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(320, 200);
    ctx.lineTo(390, 250);
    ctx.stroke();
    
    // Legs
    ctx.beginPath();
    ctx.moveTo(320, 300);
    ctx.lineTo(280, 400);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(320, 300);
    ctx.lineTo(360, 400);
    ctx.stroke();
    
    return new Promise<Blob>((resolve) => {
      canvas.toBlob((blob) => {
        if (blob) {
          resolve(blob);
        } else {
          // If toBlob fails, create a fallback blob
          const fallbackBlob = new Blob([new Uint8Array(100)], { type: 'image/jpeg' });
          resolve(fallbackBlob);
        }
      }, 'image/jpeg');
    });
  };
  
  // Train Azure Custom Vision model
  const trainAzureCustomVision = async () => {
    if (!detector) return;
    
    setTrainingStatus('Training Azure Custom Vision model...');
    
    try {
      const success = await detector.trainAzureModel(userId);
      
      if (success) {
        setTrainingStatus('Azure Custom Vision model trained successfully!');
        setUseAzureCustomVision(true);
      } else {
        setTrainingStatus('Failed to train Azure Custom Vision model.');
      }
    } catch (error) {
      console.error('Error training Azure Custom Vision:', error);
      setTrainingStatus(`Error: ${String(error)}`);
    }
  };
  
  // Train Azure ML model
  const trainAzureML = async () => {
    if (!detector) return;
    
    setTrainingStatus('Training Azure ML model...');
    
    try {
      const success = await detector.trainAzureMLModel?.(userId);
      
      if (success) {
        setTrainingStatus('Azure ML model trained successfully!');
        setUseAzureML(true);
      } else {
        setTrainingStatus('Failed to train Azure ML model.');
      }
    } catch (error) {
      console.error('Error training Azure ML:', error);
      setTrainingStatus(`Error: ${String(error)}`);
    }
  };
  
  // Handle Azure Vision toggle
  const handleAzureVisionToggle = (checked: boolean) => {
    setUseAzureVision(checked);
    // You would need to implement setUseAzureVision in your WebcamCapture component
    // or wherever the pose detection is happening
  };
  
  // Handle Azure Custom Vision toggle
  const handleAzureCustomVisionToggle = (checked: boolean) => {
    setUseAzureCustomVision(checked);
    
    if (detector && detector.setUseAzureCustomVision) {
      detector.setUseAzureCustomVision(checked);
    }
  };
  
  // Handle Azure ML toggle
  const handleAzureMLToggle = (checked: boolean) => {
    setUseAzureML(checked);
    
    if (detector && detector.setUseAzureML) {
      detector.setUseAzureML(checked);
    }
  };
  
  return (
    <div className="azure-integration-settings">
      <h2>Azure AI Integration</h2>
      <p>Enhance your experience with Azure Cognitive Services</p>
      
      <div className="azure-services">
        {/* Azure Computer Vision */}
        <div className="service-item">
          <div className="service-header">
            <h3>Azure Computer Vision</h3>
            <div className="toggle-container">
              <input 
                type="checkbox" 
                id="azure-vision-toggle"
                checked={useAzureVision}
                onChange={(e) => handleAzureVisionToggle(e.target.checked)}
              />
              <label htmlFor="azure-vision-toggle" className="toggle-label"></label>
            </div>
          </div>
          <p>Enhanced pose detection with Azure's Computer Vision API</p>
          <button 
            onClick={testAzureVision}
            disabled={!useAzureVision}
            className="azure-button"
          >
            Test Azure Vision
          </button>
        </div>
        
        {/* Azure Custom Vision */}
        <div className="service-item">
          <div className="service-header">
            <h3>Azure Custom Vision</h3>
            <div className="toggle-container">
              <input 
                type="checkbox" 
                id="azure-custom-vision-toggle"
                checked={useAzureCustomVision}
                onChange={(e) => handleAzureCustomVisionToggle(e.target.checked)}
              />
              <label htmlFor="azure-custom-vision-toggle" className="toggle-label"></label>
            </div>
          </div>
          <p>Personalized intentional movement classification</p>
          <button 
            onClick={trainAzureCustomVision}
            className="azure-button"
          >
            Train Custom Vision Model
          </button>
        </div>
        
        {/* Azure Speech Service */}
        <div className="service-item">
          <div className="service-header">
            <h3>Azure Speech Service</h3>
            <div className="toggle-container">
              <input 
                type="checkbox" 
                id="azure-speech-toggle"
                checked={useAzureSpeech}
                onChange={(e) => toggleSpeechRecognition()}
              />
              <label htmlFor="azure-speech-toggle" className="toggle-label"></label>
            </div>
          </div>
          <p>Voice commands for hands-free control</p>
          <div className="voice-commands">
            <p>Available commands:</p>
            <ul>
              <li>start, stop</li>
              <li>piano, drums, guitar</li>
              <li>louder, softer</li>
              <li>faster, slower</li>
            </ul>
          </div>
        </div>
        
        {/* Azure Machine Learning */}
        <div className="service-item">
          <div className="service-header">
            <h3>Azure Machine Learning</h3>
            <div className="toggle-container">
              <input 
                type="checkbox" 
                id="azure-ml-toggle"
                checked={useAzureML}
                onChange={(e) => handleAzureMLToggle(e.target.checked)}
              />
              <label htmlFor="azure-ml-toggle" className="toggle-label"></label>
            </div>
          </div>
          <p>Advanced movement pattern analysis and filtering</p>
          <button 
            onClick={trainAzureML}
            className="azure-button"
          >
            Train ML Model
          </button>
        </div>
      </div>
      
      {/* Status display */}
      {(trainingStatus || azureTestResponse) && (
        <div className="azure-status">
          <h3>Azure Services Status</h3>
          {trainingStatus && <p className="training-status">{trainingStatus}</p>}
          {azureTestResponse && <p className="test-response">{azureTestResponse}</p>}
        </div>
      )}
    </div>
  );
};

export default AzureIntegrationSettings;