import React, { useRef, useEffect, useState } from 'react';

interface SoundVisualizerProps {
  soundEngine: any; // Use SoundEngine type if available
  isActive: boolean;
}

const SoundVisualizer: React.FC<SoundVisualizerProps> = ({ soundEngine, isActive }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [barValues, setBarValues] = useState<number[]>(Array(16).fill(0));
  const animationRef = useRef<number | null>(null);
  
  useEffect(() => {
    if (!isActive) {
      // Reset bars when inactive
      setBarValues(Array(16).fill(0));
      return;
    }
    
    // Animation function to update bars
    const animate = () => {
      // Generate random values when sound is active
      // In a real implementation, you could connect this to actual audio analysis
      const newValues = barValues.map(prev => {
        // Calculate a new value that sometimes goes up, sometimes down
        let newVal = prev + (Math.random() * 0.4 - 0.2);
        
        // Ensure values stay within bounds
        newVal = Math.max(0, Math.min(1, newVal));
        
        return newVal;
      });
      
      setBarValues(newValues);
      animationRef.current = requestAnimationFrame(animate);
    };
    
    // Start animation
    animationRef.current = requestAnimationFrame(animate);
    
    // Cleanup
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isActive]);
  
  // Draw bars on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw each bar
    const barWidth = canvas.width / barValues.length;
    const heightMultiplier = canvas.height * 0.8;
    
    barValues.forEach((value, index) => {
      const x = index * barWidth;
      const height = value * heightMultiplier;
      const y = canvas.height - height;
      
      // Create gradient
      const gradient = ctx.createLinearGradient(0, canvas.height, 0, 0);
      gradient.addColorStop(0, '#3498db');
      gradient.addColorStop(0.6, '#2ecc71');
      gradient.addColorStop(1, '#f1c40f');
      
      // Draw the bar
      ctx.fillStyle = gradient;
      ctx.fillRect(x, y, barWidth - 2, height);
      
      // Add glow effect for active visualization
      if (isActive && value > 0.5) {
        ctx.shadowColor = '#3498db';
        ctx.shadowBlur = 10;
        ctx.fillStyle = 'rgba(52, 152, 219, 0.5)';
        ctx.fillRect(x, y, barWidth - 2, height);
        ctx.shadowBlur = 0;
      }
    });
    
  }, [barValues, isActive]);
  
  return (
    <div className="sound-visualizer-container">
      <canvas
        ref={canvasRef}
        width={300}
        height={60}
        className={`sound-visualizer ${isActive ? 'active' : 'inactive'}`}
      />
      <div className="visualizer-label">
        {isActive ? 'Sound Active' : 'Sound Inactive'}
      </div>
    </div>
  );
};

export default SoundVisualizer;