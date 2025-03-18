import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs';
import { 
  MovementFeatures, 
  MovementInfo, 
  CalibrationSample,
  AzureServiceConfig,
  DetectorStatus,
  TRACKED_KEYPOINTS
} from './DetectionTypes';

/**
 * Compatibility wrapper for MLIntentionDetector
 * 
 * This provides compatibility with the original API while
 * gradually transitioning to the improved architecture.
 * 
 * Place this file at src/utils/IntentionDetection/MLIntentionDetector.ts
 * to maintain compatibility with existing imports.
 */
export class MLIntentionDetector {
  // Basic state
  private posesHistory: Array<{poses: poseDetection.Pose[], timestamp: number}> = [];
  private lastIntentionalMovements = new Map<string, number>();
  private calibrationSamples: CalibrationSample[] = [];
  private calibrationQuality = 0;
  private isModelTrained = false;
  private useMLPrediction = false;
  private calibrationDuration = 15;
  private cooldownPeriod = 200;
  private minConfidence = 0.5;
  
  // Azure configuration
  private azureConfig: AzureServiceConfig = {
    apiKey: '',
    endpoint: '',
    enabled: false
  };
  
  // Error tracking
  private lastError = '';
  private errorCount = 0;
  
  // Training callbacks
  private trainingProgressCallback: ((progress: number) => void) | null = null;
  private isAzureTraining = false;
  
  constructor() {
    // Initialize TensorFlow
    this.initializeTensorflow()
      .then(() => console.log('TensorFlow initialized'))
      .catch(error => console.error('TensorFlow initialization failed:', error));
    
    console.log('MLIntentionDetector compatibility wrapper initialized');
  }
  
  /**
   * Initialize TensorFlow.js
   */
  private async initializeTensorflow(): Promise<void> {
    try {
      await tf.ready();
      
      // Try to set WebGL backend
      try {
        await tf.setBackend('webgl');
        console.log('Using WebGL backend');
        
        // Optimize WebGL
        tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0);
        tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
        tf.env().set('WEBGL_FLUSH_THRESHOLD', 1);
      } catch (error) {
        console.warn('WebGL backend failed, trying CPU:', error);
        await tf.setBackend('cpu');
      }
    } catch (error) {
      this.lastError = `TensorFlow initialization error: ${error}`;
      this.errorCount++;
      throw new Error(this.lastError);
    }
  }
  
  /**
   * Process pose data to detect intentional movements
   */
  public processPoses(poses: poseDetection.Pose[]): MovementInfo[] {
    // Add to history
    this.posesHistory.push({
      poses,
      timestamp: Date.now()
    });
    
    // Trim history if it gets too long
    while (this.posesHistory.length > 60) {
      this.posesHistory.shift();
    }
    
    // Need a few frames to detect movement
    if (this.posesHistory.length < 10) {
      return [];
    }
    
    const movements: MovementInfo[] = [];
    
    // No poses? Return empty
    if (poses.length === 0) {
      return movements;
    }
    
    // Process keypoints using heuristic approach
    for (const keypointName of TRACKED_KEYPOINTS) {
      const keypoint = this.findKeypointByName(poses[0], keypointName);
      
      if (keypoint && keypoint.score && keypoint.score > this.minConfidence) {
        // Extract basic velocity from last few frames
        const oldestPose = this.posesHistory[0].poses[0];
        const oldKeypoint = this.findKeypointByName(oldestPose, keypointName);
        
        if (!oldKeypoint || !keypoint.x || !oldKeypoint.x) continue;
        
        // Calculate velocity and direction
        const dx = keypoint.x - oldKeypoint.x;
        const dy = keypoint.y - oldKeypoint.y;
        const velocity = Math.sqrt(dx * dx + dy * dy);
        
        // Simple direction
        let direction = 'none';
        if (Math.abs(dx) > Math.abs(dy)) {
          direction = dx > 0 ? 'right' : 'left';
        } else {
          direction = dy > 0 ? 'down' : 'up';
        }
        
        // Simple heuristic for intentionality
        const isIntentional = velocity > 15;
        
        // Apply cooldown
        const notOnCooldown = this.isNotOnCooldown(keypointName);
        const finalIntentional = isIntentional && notOnCooldown;
        
        if (finalIntentional) {
          this.lastIntentionalMovements.set(keypointName, Date.now());
        }
        
        if (finalIntentional || velocity > 5) {
          movements.push({
            keypoint: keypointName,
            isIntentional: finalIntentional,
            velocity,
            direction,
            confidence: finalIntentional ? 0.8 : 0.3
          });
        }
      }
    }
    
    return movements;
  }
  
  /**
   * Check cooldown period to prevent rapid triggers
   */
  private isNotOnCooldown(keypointName: string): boolean {
    const lastMovement = this.lastIntentionalMovements.get(keypointName);
    if (!lastMovement) return true;
    
    const now = Date.now();
    return (now - lastMovement) > this.cooldownPeriod;
  }
  
  /**
   * Helper to find a keypoint by name in pose data
   */
  private findKeypointByName(pose: poseDetection.Pose, name: string): poseDetection.Keypoint | undefined {
    return pose.keypoints.find(kp => kp.name === name);
  }
  
  /**
   * Add a sample during calibration
   */
  public addCalibrationSample(isIntentional: boolean): void {
    // Simple placeholder implementation that doesn't actually extract features
    // Just to maintain API compatibility
    
    const timestamp = Date.now();
    const samplesAdded = Math.min(TRACKED_KEYPOINTS.length, 3);
    
    // Add a few dummy samples
    for (let i = 0; i < samplesAdded; i++) {
      // Use the keypoints with the highest predicted presence
      const keypoint = TRACKED_KEYPOINTS[i];
      
      // Create dummy features
      const features: MovementFeatures = {
        keypoint,
        velocityX: Math.random() * 10,
        velocityY: Math.random() * 10,
        acceleration: Math.random() * 5,
        jitter: isIntentional ? 5 : 15,
        direction: ['up', 'down', 'left', 'right'][Math.floor(Math.random() * 4)],
        isSmooth: isIntentional,
        timestamp,
        magnitudeOfMovement: isIntentional ? 20 : 5,
        durationOfMovement: 0.5,
        isReversing: !isIntentional,
        frequencyOfMovement: isIntentional ? 1 : 5,
        steadiness: isIntentional ? 0.8 : 0.3,
        patternScore: isIntentional ? 0.7 : 0.2,
        continuity: isIntentional ? 0.9 : 0.4
      };
      
      this.calibrationSamples.push({
        features,
        isIntentional
      });
    }
    
    // Log calibration progress occasionally
    if (this.calibrationSamples.length % 20 === 0) {
      const intentionalCount = this.calibrationSamples.filter(s => s.isIntentional).length;
      const unintentionalCount = this.calibrationSamples.length - intentionalCount;
      
      console.log(`Calibration samples: ${this.calibrationSamples.length} (${intentionalCount} intentional, ${unintentionalCount} unintentional)`);
    }
  }
  
  /**
   * Train model with collected calibration samples
   */
  public async trainModel(): Promise<boolean> {
    console.log(`Training model with ${this.calibrationSamples.length} samples`);
    
    // Simulate training
    for (let i = 0; i < 10; i++) {
      if (this.trainingProgressCallback) {
        this.trainingProgressCallback(i / 10);
      }
      await new Promise(resolve => setTimeout(resolve, 200));
    }
    
    // Final progress
    if (this.trainingProgressCallback) {
      this.trainingProgressCallback(1.0);
    }
    
    // Mark as trained
    this.isModelTrained = true;
    this.useMLPrediction = true;
    
    // Calculate dummy quality score
    const intentionalSamples = this.calibrationSamples.filter(s => s.isIntentional).length;
    const unintentionalSamples = this.calibrationSamples.filter(s => !s.isIntentional).length;
    const balance = Math.min(intentionalSamples, unintentionalSamples) / 
                    Math.max(intentionalSamples, unintentionalSamples);
    
    this.calibrationQuality = Math.round(balance * 80);
    
    return true;
  }
  
  /**
   * Set callback for training progress updates
   */
  public setTrainingProgressCallback(callback: (progress: number) => void): void {
    this.trainingProgressCallback = callback;
  }
  
  /**
   * Configure Azure cognitive services
   */
  public configureAzureServices(config: AzureServiceConfig): void {
    this.azureConfig = {
      ...this.azureConfig,
      ...config
    };
  }
  
  /**
   * Set calibration duration
   */
  public setCalibrationDuration(seconds: number): void {
    this.calibrationDuration = Math.max(5, seconds);
  }
  
  /**
   * Get calibration duration
   */
  public getCalibrationDuration(): number {
    return this.calibrationDuration;
  }
  
  /**
   * Clear calibration data
   */
  public clearCalibration(): void {
    this.calibrationSamples = [];
    this.calibrationQuality = 0;
    this.lastIntentionalMovements.clear();
    console.log('Calibration samples cleared');
  }
  
  /**
   * Get the status of the detector
   */
  public getStatus(): DetectorStatus {
    const intentionalSamples = this.calibrationSamples.filter(s => s.isIntentional).length;
    
    return {
      isModelTrained: this.isModelTrained,
      calibrationSamples: this.calibrationSamples.length,
      intentionalSamples,
      unintentionalSamples: this.calibrationSamples.length - intentionalSamples,
      isTfInitialized: true,
      calibrationQuality: this.calibrationQuality,
      azureEnabled: this.azureConfig.enabled,
      lastError: this.lastError,
      isUsingWorkers: false // Compatibility layer doesn't use workers
    };
  }
  
  /**
   * Train Azure model - compatibility method
   */
  public async trainAzureModel(userId: string): Promise<boolean> {
    if (!this.azureConfig.enabled) {
      console.warn('Azure integration not enabled');
      return false;
    }
    
    this.isAzureTraining = true;
    
    try {
      // Simulate Azure training
      for (let i = 0; i < 20; i++) {
        if (this.trainingProgressCallback) {
          this.trainingProgressCallback(i / 20);
        }
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      
      // Final progress
      if (this.trainingProgressCallback) {
        this.trainingProgressCallback(1.0);
      }
      
      this.isAzureTraining = false;
      
      // Simulate creating a model ID
      const modelId = `model-${userId}-${Date.now()}`;
      this.azureConfig.defaultModelId = modelId;
      
      // Store in local storage to simulate persistence
      localStorage.setItem(`azure-model-id-${userId}`, modelId);
      
      return true;
    } catch (error) {
      this.lastError = `Error training Azure model: ${error}`;
      this.errorCount++;
      console.error(this.lastError);
      this.isAzureTraining = false;
      return false;
    }
  }
  
  /**
   * Save the model for this user
   */
  public async saveModel(userId: string): Promise<boolean> {
    if (!this.isModelTrained) {
      console.warn('No trained model to save');
      return false;
    }
    
    try {
      // Simulate saving
      console.log(`Saving model for user ${userId}`);
      
      // Store calibration quality
      localStorage.setItem(`user-calibration-quality-${userId}`, this.calibrationQuality.toString());
      
      // Create a dummy model representation
      const modelData = {
        userId,
        timestamp: Date.now(),
        samples: this.calibrationSamples.length,
        quality: this.calibrationQuality
      };
      
      // Store in local storage
      localStorage.setItem(`user-model-data-${userId}`, JSON.stringify(modelData));
      
      return true;
    } catch (error) {
      this.lastError = `Error saving model: ${error}`;
      this.errorCount++;
      console.error(this.lastError);
      return false;
    }
  }
  
  /**
   * Load a saved model for this user
   */
  public async loadModel(userId: string): Promise<boolean> {
    try {
      console.log(`Loading model for user ${userId}`);
      
      // Try to load calibration quality
      const qualityStr = localStorage.getItem(`user-calibration-quality-${userId}`);
      if (qualityStr) {
        this.calibrationQuality = parseInt(qualityStr, 10);
      }
      
      // Check if model data exists
      const modelDataStr = localStorage.getItem(`user-model-data-${userId}`);
      if (!modelDataStr) {
        console.warn(`No saved model found for user ${userId}`);
        return false;
      }
      
      // Parse model data
      const modelData = JSON.parse(modelDataStr);
      
      // Simulate loading the model
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Mark as trained
      this.isModelTrained = true;
      this.useMLPrediction = true;
      
      console.log(`Loaded model for user ${userId} (quality: ${this.calibrationQuality})`);
      return true;
    } catch (error) {
      this.lastError = `Error loading model: ${error}`;
      this.errorCount++;
      console.error(this.lastError);
      return false;
    }
  }
  
  /**
   * Clean up resources
   */
  public dispose(): void {
    // Clean up any resources if needed
    console.log('MLIntentionDetector: Disposed');
  }
}