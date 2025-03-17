import React, { useState, useEffect, useRef } from 'react';
import { MovementInfo } from '../utils/MLIntentionDetector';
import './GraphicScoreComponent.css';

interface GraphicScoreProps {
  recordingEnabled: boolean;
  onToggleRecording: () => void;
  recordedMovements: MovementInfo[];
  showDebugInfo?: boolean;
}

const GraphicScoreComponent: React.FC<GraphicScoreProps> = ({ 
  recordingEnabled, 
  onToggleRecording, 
  recordedMovements,
  showDebugInfo = false
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [notationType, setNotationType] = useState<'abstract' | 'traditional'>('abstract');
  const [autoScroll, setAutoScroll] = useState<boolean>(true);
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  
  // Colors for different keypoints
  const keypointColors: {[key: string]: string} = {
    'right_wrist': '#FF5722',
    'left_wrist': '#2196F3',
    'right_elbow': '#FFC107',
    'left_elbow': '#4CAF50',
    'nose': '#9C27B0',
    'right_shoulder': '#FF9800',
    'left_shoulder': '#00BCD4',
    'right_index': '#FFEB3B',
    'left_index': '#8BC34A',
    'right_thumb': '#E91E63',
    'left_thumb': '#03A9F4',
    'right_pinky': '#CDDC39',
    'left_pinky': '#009688',
    'right_eye': '#F44336',
    'left_eye': '#3F51B5'
  };
  
  // Direction symbols for traditional notation
  const directionSymbols: {[key: string]: string} = {
    'up': '↑',
    'down': '↓',
    'left': '←',
    'right': '→'
  };

  // Draw the score on the canvas
  useEffect(() => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Set background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw time grid
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    
    const gridSpacing = 50 * zoomLevel;
    for (let x = 0; x < canvas.width; x += gridSpacing) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    
    // Get list of keypoints from recorded movements
    const keypoints = Array.from(
      new Set(recordedMovements.map(m => m.keypoint))
    ).filter(k => keypointColors[k]); // Only include keypoints we have colors for
    
    // Use all known keypoints if we don't have any recorded movements
    if (keypoints.length === 0) {
      Object.keys(keypointColors).forEach(k => keypoints.push(k));
    }
    
    // Draw horizontal lines for keypoint areas
    const keypointHeight = canvas.height / keypoints.length;
    
    keypoints.forEach((keypoint, index) => {
      const y = index * keypointHeight;
      
      // Draw keypoint area
      ctx.fillStyle = `${keypointColors[keypoint]}20`; // 20 = 12% opacity
      ctx.fillRect(0, y, canvas.width, keypointHeight);
      
      // Draw keypoint label
      ctx.fillStyle = keypointColors[keypoint];
      ctx.font = '12px Arial';
      ctx.fillText(keypoint.replace('_', ' '), 5, y + 15);
      
      // Draw horizontal line
      ctx.strokeStyle = '#e0e0e0';
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    });
    
    // Draw recorded movements
    if (recordedMovements.length > 0) {
      const timeScale = 10 * zoomLevel; // Pixels per time unit
      
      recordedMovements.forEach((movement, index) => {
        const keypointIndex = keypoints.indexOf(movement.keypoint);
        
        if (keypointIndex !== -1) {
          const x = index * timeScale;
          const y = keypointIndex * keypointHeight + (keypointHeight / 2);
          
          // Only draw if within canvas bounds
          if (x < canvas.width) {
            if (notationType === 'abstract') {
              // Draw abstract notation
              // Circle size based on velocity
              const radius = Math.min(20, Math.max(5, movement.velocity / 2));
              
              ctx.beginPath();
              ctx.arc(x, y, radius, 0, Math.PI * 2);
              
              // Fill based on intentionality
              if (movement.isIntentional) {
                ctx.fillStyle = keypointColors[movement.keypoint];
              } else {
                ctx.fillStyle = '#cccccc';
              }
              
              ctx.fill();
              
              // Draw direction line
              if (movement.direction !== 'none') {
                ctx.strokeStyle = movement.isIntentional ? '#333333' : '#999999';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(x, y);
                
                const lineLength = Math.min(30, Math.max(10, movement.velocity));
                
                switch (movement.direction) {
                  case 'up':
                    ctx.lineTo(x, y - lineLength);
                    break;
                  case 'down':
                    ctx.lineTo(x, y + lineLength);
                    break;
                  case 'left':
                    ctx.lineTo(x - lineLength, y);
                    break;
                  case 'right':
                    ctx.lineTo(x + lineLength, y);
                    break;
                }
                
                ctx.stroke();
              }
            } else {
              // Draw traditional-inspired notation
              ctx.font = `${Math.min(30, Math.max(15, movement.velocity / 2))}px Arial`;
              ctx.fillStyle = movement.isIntentional ? keypointColors[movement.keypoint] : '#cccccc';
              ctx.fillText(directionSymbols[movement.direction] || '•', x, y);
            }
          }
        }
      });
      
      // Auto-scroll if enabled and movements extend beyond canvas
      if (autoScroll && recordedMovements.length * timeScale > canvas.width) {
        const scrollX = (recordedMovements.length * timeScale) - canvas.width;
        canvas.style.transform = `translateX(-${scrollX}px)`;
      }
    }
  }, [recordedMovements, notationType, zoomLevel, autoScroll, keypointColors]);

  const handleSaveImage = () => {
    if (canvasRef.current) {
      // Create a temporary link element
      const link = document.createElement('a');
      link.download = 'movement-score.png';
      link.href = canvasRef.current.toDataURL('image/png');
      link.click();
    }
  };

  return (
    <div className="graphic-score-container">
      <div className="score-controls">
        <h3>Graphic Score</h3>
        
        <div className="control-group">
          <button 
            className={`record-button ${recordingEnabled ? 'recording' : ''}`}
            onClick={onToggleRecording}
          >
            {recordingEnabled ? 'Stop Recording' : 'Start Recording'}
          </button>
          
          <button 
            className="save-button"
            onClick={handleSaveImage}
            disabled={recordedMovements.length === 0}
          >
            Save Image
          </button>
        </div>
        
        <div className="control-group">
          <label>
            Notation Type:
            <select 
              value={notationType} 
              onChange={(e) => setNotationType(e.target.value as 'abstract' | 'traditional')}
            >
              <option value="abstract">Abstract</option>
              <option value="traditional">Traditional</option>
            </select>
          </label>
          
          <label>
            <input 
              type="checkbox" 
              checked={autoScroll} 
              onChange={(e) => setAutoScroll(e.target.checked)}
            />
            Auto-scroll
          </label>
        </div>
        
        <div className="control-group">
          <label>
            Zoom:
            <input 
              type="range" 
              min="0.5" 
              max="2" 
              step="0.1" 
              value={zoomLevel}
              onChange={(e) => setZoomLevel(parseFloat(e.target.value))}
            />
          </label>
        </div>
      </div>
      
      <div className="score-canvas-container">
        <canvas 
          ref={canvasRef} 
          width={1200} 
          height={600}
          className="score-canvas"
        />
      </div>
      
      {showDebugInfo && recordedMovements.length > 0 && (
        <div className="score-debug-info">
          <h4>Recorded Movements: {recordedMovements.length}</h4>
          <ul className="movement-list">
            {recordedMovements.slice(-5).map((movement, idx) => (
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
  );
};

export default GraphicScoreComponent;