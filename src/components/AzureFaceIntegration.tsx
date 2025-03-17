// src/components/AzureFaceIntegration.tsx

import React, { useState } from 'react';
import { AzureFaceService } from '../services/azureFaceService';


interface AzureFaceIntegrationProps {
  onFaceDataReceived?: (faceData: any) => void;
}

const AzureFaceIntegration: React.FC<AzureFaceIntegrationProps> = ({ 
  onFaceDataReceived 
}) => {
  const [useAzureFace, setUseAzureFace] = useState<boolean>(false);
  const [faceService] = useState<AzureFaceService>(new AzureFaceService());
  const [testResponse, setTestResponse] = useState<string>('');
  const [testImageUrl, setTestImageUrl] = useState<string | null>(null);
  
  // Test Azure Face API
  const testAzureFace = async () => {
    try {
      setTestResponse('Testing Azure Face API...');
      setTestImageUrl(null);
      
      // Get a sample image for testing
      const testImage = await getSampleImageBlob();
      
      // Display the test image
      const imageUrl = URL.createObjectURL(testImage);
      setTestImageUrl(imageUrl);
      
      // Detect faces
      const detectedFaces = await faceService.detectFace(testImage);
      
      if (detectedFaces && detectedFaces.length > 0) {
        const face = detectedFaces[0];
        const emotion = faceService.getDominantEmotion(face);
        const headPose = faceService.getHeadPose(face);
        
        setTestResponse(`
          Face detected successfully!
          - Face ID: ${face.faceId}
          - Position: ${face.faceRectangle.left},${face.faceRectangle.top}
          - Size: ${face.faceRectangle.width} x ${face.faceRectangle.height}
          - Dominant emotion: ${emotion || 'unknown'}
          - Head pose: Pitch ${headPose?.pitch.toFixed(2) || 'N/A'}, Yaw ${headPose?.yaw.toFixed(2) || 'N/A'}
        `);
        
        if (onFaceDataReceived) {
          onFaceDataReceived(face);
        }
      } else {
        setTestResponse('No faces detected in the test image.');
      }
    } catch (error) {
      console.error('Error testing Azure Face API:', error);
      setTestResponse(`Azure Face API error: ${String(error)}`);
    }
  };
  
  // Helper function to get a sample image for testing
  const getSampleImageBlob = async (): Promise<Blob> => {
    // Create a canvas and draw a simple face
    const canvas = document.createElement('canvas');
    canvas.width = 400;
    canvas.height = 400;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Could not get canvas context');
    
    // Background
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw a simple face
    // Head
    ctx.beginPath();
    ctx.arc(200, 200, 150, 0, Math.PI * 2);
    ctx.fillStyle = '#f0d0b0';
    ctx.fill();
    
    // Eyes
    ctx.beginPath();
    ctx.arc(150, 150, 20, 0, Math.PI * 2);
    ctx.fillStyle = 'white';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(250, 150, 20, 0, Math.PI * 2);
    ctx.fillStyle = 'white';
    ctx.fill();
    
    // Pupils
    ctx.beginPath();
    ctx.arc(150, 150, 8, 0, Math.PI * 2);
    ctx.fillStyle = 'black';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(250, 150, 8, 0, Math.PI * 2);
    ctx.fillStyle = 'black';
    ctx.fill();
    
    // Mouth
    ctx.beginPath();
    ctx.arc(200, 250, 50, 0, Math.PI);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 3;
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
  
  // Handle toggle
  const handleAzureFaceToggle = (checked: boolean) => {
    setUseAzureFace(checked);
    // Additional logic to enable/disable face detection in your application
  };
  
  return (
    <div className="service-item">
      <div className="service-header">
        <h3>Azure Face API</h3>
        <div className="toggle-container">
          <input 
            type="checkbox" 
            id="azure-face-toggle"
            checked={useAzureFace}
            onChange={(e) => handleAzureFaceToggle(e.target.checked)}
          />
          <label htmlFor="azure-face-toggle" className="toggle-label"></label>
        </div>
      </div>
      <p>Advanced facial expression and head pose detection</p>
      <button 
        onClick={testAzureFace}
        disabled={!useAzureFace}
        className="azure-button"
      >
        Test Face API
      </button>
      
      {testImageUrl && (
        <div className="test-image-container">
          <img 
            src={testImageUrl} 
            alt="Test face" 
            style={{ 
              maxWidth: '200px', 
              maxHeight: '200px',
              marginTop: '10px',
              border: '1px solid #ccc'
            }} 
          />
        </div>
      )}
      
      {testResponse && (
        <div className="test-response" style={{ whiteSpace: 'pre-line' }}>
          {testResponse}
        </div>
      )}
    </div>
  );
};

export default AzureFaceIntegration;