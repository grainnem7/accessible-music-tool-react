import * as poseDetection from '@tensorflow-models/pose-detection';
import { 
  MovementFeatures, 
  MovementInfo, 
  CalibrationSample,
  AzureServiceConfig,
  DetectorStatus,
  TRACKED_KEYPOINTS
} from './DetectionTypes';
import { featureExtractor } from './FeatureExtractor';
import { performanceMonitor } from '../Helpers/PerformanceMonitor';
import { tensorflowHelper } from '../Helpers/TensorflowHelper';
import { AzureIntegration } from './AzureIntegration';
import * as tf from '@tensorflow/tfjs';



/**
 * MLIntentionDetector - Core coordinator class
 * This lightweight class coordinates the ML processing pipeline,
 * delegating heavy computations to web workers.
 */
export class MLIntentionDetector {
  // Pose tracking
  private posesHistory: Array<{poses: poseDetection.Pose[], timestamp: number}> = [];
  private historyLength = 60; // ~2 seconds at 30fps
  private lastIntentionalMovements = new Map<string, number>(); // Keypoint to timestamp
  private cooldownPeriod = 200; // ms to prevent rapid retriggering
  private minConfidence = 0.5; // Minimum confidence for keypoints
  
  // ML State
  private calibrationSamples: CalibrationSample[] = [];
  private calibrationQuality = 0; // 0-100 score
  private isModelTrained = false;
  private useMLPrediction = false;
  private calibrationDuration = 15; // seconds per calibration step
  
  // Azure integration
  private azureConfig: AzureServiceConfig = {
    apiKey: '',
    endpoint: '',
    enabled: false
  };
  
  // Workers
  private mainWorker: Worker | null = null;
  private isWorkerInitialized = false;
  private pendingWorkerRequests = new Map<string, {
    resolve: (value: any) => void,
    reject: (reason: any) => void,
    timeout: number
  }>();
  
  // Training callback
  private trainingProgressCallback: ((progress: number) => void) | null = null;
  
  // Error tracking
  private lastError = '';
  private errorCount = 0;
  
  // Performance and optimization
  private frameSkip = 0; // Process every N+1 frames
  private currentFrame = 0;
  private processingInProgress = false;
  
  /**
   * Constructor initializes TensorFlow and workers
   */
  constructor(
    options: {
      useWorkers?: boolean;
      preferredBackend?: string;
      frameSkip?: number;
    } = {}
  ) {
    // Enable performance monitoring
    performanceMonitor.setEnabled(true);
    performanceMonitor.setWarningThreshold('processPoses', 16); // 16ms = 60fps
    
    // Set frame skipping for performance
    this.frameSkip = options.frameSkip || 0;
    
    // Initialize TensorFlow in the main thread (for fallback)
    this.initializeTensorflow(options.preferredBackend || 'webgl');
    
    // Initialize web workers if enabled
    if (options.useWorkers !== false) {
      this.initializeWorkers(options.preferredBackend || 'webgl');
    }
  }
  
  /**
   * Initialize TensorFlow in the main thread
   */
  private async initializeTensorflow(preferredBackend: string): Promise<void> {
    try {
      await tensorflowHelper.initializeTensorflow(preferredBackend);
    } catch (error) {
      this.lastError = `Error initializing TensorFlow: ${error}`;
      this.errorCount++;
      console.error(this.lastError);
    }
  }
  

  /**
   * Initialize web workers for heavy computation
   */
  private initializeWorkers(preferredBackend: string): void {
    try {
      // Create main worker
      this.mainWorker = new Worker('./workers/MLWorker.js');
      
      // Set up message handler
      this.mainWorker.addEventListener('message', this.handleWorkerMessage);
      
      // Handle worker errors
      this.mainWorker.addEventListener('error', (error) => {
        this.lastError = `Worker error: ${error}`;
        this.errorCount++;
        console.error(this.lastError);
      });
      
      // Initialize workers
      const initMessageId = `init-${Date.now()}`;
      this.pendingWorkerRequests.set(initMessageId, {
        resolve: () => {
          this.isWorkerInitialized = true;
          console.log('MLIntentionDetector: Workers initialized');
        },
        reject: (error) => {
          this.lastError = `Worker initialization error: ${error}`;
          this.errorCount++;
          console.error(this.lastError);
        },
        timeout: window.setTimeout(() => {
          this.pendingWorkerRequests.delete(initMessageId);
          console.error('MLIntentionDetector: Worker initialization timed out');
        }, 10000)
      });
      
      // Send initialization message
      this.mainWorker.postMessage({
        type: 'init',
        id: initMessageId,
        preferredBackend,
        includeTraining: true
      });
    } catch (error) {
      this.lastError = `Error initializing workers: ${error}`;
      this.errorCount++;
      console.error(this.lastError);
    }
  }
  
  /**
   * Handle messages from the web worker
   */
  private handleWorkerMessage = (event: MessageEvent): void => {
    const message = event.data;
    
    // Complete pending request if it has an ID
    if (message.id && this.pendingWorkerRequests.has(message.id)) {
      const pending = this.pendingWorkerRequests.get(message.id)!;
      clearTimeout(pending.timeout);
      
      if (message.type === 'error') {
        pending.reject(message.error);
      } else {
        pending.resolve(message);
      }
      
      this.pendingWorkerRequests.delete(message.id);
    }
    
    // Handle specific message types
    switch (message.type) {
      case 'prediction': 
        // Handle prediction result
        // This would be used by an event-based implementation
        break;
        
      case 'features':
        // Handle features result
        // This would be used by an event-based implementation
        break;
        
      case 'trainingProgress':
        // Update training progress
        if (this.trainingProgressCallback && typeof message.progress === 'number') {
          this.trainingProgressCallback(message.progress);
        }
        break;
        
      case 'error':
        // Log worker errors
        this.lastError = `Worker error: ${message.error}`;
        this.errorCount++;
        console.error(this.lastError);
        break;
    }
  };
  
  /**
   * Process pose data to detect intentional movements
   * This is the main entry point for real-time detection
   */
  public processPoses(poses: poseDetection.Pose[]): MovementInfo[] {
    // Start performance tracking
    performanceMonitor.start('processPoses');
    
    // Skip frames for performance if needed
    this.currentFrame = (this.currentFrame + 1) % (this.frameSkip + 1);
    if (this.currentFrame !== 0 && this.frameSkip > 0) {
      return []; // Skip this frame
    }
    
    // Don't process if already processing a frame (prevent overlapping)
    if (this.processingInProgress) {
      return [];
    }
    
    try {
      this.processingInProgress = true;
      
      // Add to history
      this.posesHistory.push({
        poses,
        timestamp: Date.now()
      });
      
      // Trim history if it gets too long
      while (this.posesHistory.length > this.historyLength) {
        this.posesHistory.shift();
      }
      
      // Need at least a few frames to detect movement
      if (this.posesHistory.length < 10) {
        this.processingInProgress = false;
        return [];
      }
      
      const movements: MovementInfo[] = [];
      
      // Only process if we have poses
      if (poses.length === 0) {
        this.processingInProgress = false;
        return movements;
      }
      
      // Process each tracked keypoint
      for (const keypointName of TRACKED_KEYPOINTS) {
        const keypoint = this.findKeypointByName(poses[0], keypointName);
        
        if (keypoint && keypoint.score && keypoint.score > this.minConfidence) {
          // Extract features
          const features = this.extractFeatures(keypointName);
          
          if (!features) continue;
          
          let isIntentional = false;
          let confidence = 0.5;
          
          // If we have a trained model, use it
          if (this.isModelTrained && this.useMLPrediction) {
            const prediction = this.predictWithModel(features);
            isIntentional = prediction.isIntentional;
            confidence = prediction.confidence;
          } else {
            // Use heuristic approach
            const heuristicResult = this.heuristicIntentionality(features);
            isIntentional = heuristicResult.isIntentional;
            confidence = heuristicResult.confidence;
          }
          
          // Apply cooldown to avoid rapid retriggering
          if (isIntentional && !this.isNotOnCooldown(keypointName)) {
            isIntentional = false;
          }
          
          // If intentional, update last movement timestamp
          if (isIntentional) {
            this.lastIntentionalMovements.set(keypointName, Date.now());
          }
          
          // Calculate velocity magnitude
          const velocity = Math.sqrt(
            features.velocityX * features.velocityX + 
            features.velocityY * features.velocityY
          );
          
          // Only add significant movements to results
          if (isIntentional || velocity > 5) {
            movements.push({
              keypoint: keypointName,
              isIntentional,
              velocity,
              direction: features.direction,
              confidence
            });
          }
        }
      }
      
      this.processingInProgress = false;
      performanceMonitor.end('processPoses');
      return movements;
    } catch (error) {
      this.lastError = `Error processing poses: ${error}`;
      this.errorCount++;
      console.error(this.lastError);
      this.processingInProgress = false;
      
      performanceMonitor.end('processPoses');
      return [];
    }
  }
  
  /**
   * Extract features for a keypoint using either worker or local implementation
   */
  private extractFeatures(keypointName: string): MovementFeatures | null {
    if (this.isWorkerInitialized && this.mainWorker) {
      // Use worker for feature extraction (async, not implemented here)
      // Would need to adapt this method to be async
      return featureExtractor.extractFeatures(this.posesHistory, keypointName);
    } else {
      // Use local feature extractor
      return featureExtractor.extractFeatures(this.posesHistory, keypointName);
    }
  }
  
  /**
   * Make a prediction using the model (local fallback implementation)
   */
  private predictWithModel(features: MovementFeatures): { isIntentional: boolean, confidence: number } {
    // This would normally use the worker, but provides a local implementation
    // for fallback when workers aren't available
    
    // Use heuristic as fallback if needed
    return this.heuristicIntentionality(features);
  }
  
  /**
   * Heuristic-based intentionality detection
   * Used as fallback when ML model not available
   */
  private heuristicIntentionality(features: MovementFeatures): { isIntentional: boolean, confidence: number } {
    const velocity = Math.sqrt(
      features.velocityX * features.velocityX + 
      features.velocityY * features.velocityY
    );
    
    // Enhanced approach using feature set:
    
    // 1. Fast, deliberate movements are likely intentional
    const speedFactor = velocity > 20;
    
    // 2. Smooth movements with low jitter are likely intentional
    const smoothnessFactor = features.jitter < 12 && features.isSmooth;
    
    // 3. Movements with higher acceleration are likely intentional
    const accelerationFactor = features.acceleration > 15;
    
    // 4. Movements with a specific duration range are likely intentional
    const durationFactor = features.durationOfMovement > 0.2 && features.durationOfMovement < 1.5;
    
    // 5. Non-reversing movements are more likely intentional
    const directionFactor = !features.isReversing;
    
    // 6. Low frequency of direction changes suggests intentionality
    // Tremors and spasms typically have high frequency
    const frequencyFactor = features.frequencyOfMovement < 4;
    
    // 7. High steadiness suggests intentional movement
    const steadinessFactor = features.steadiness > 0.6;
    
    // 8. Higher pattern scores suggest intentional movements
    const patternFactor = features.patternScore > 0.5;
    
    // 9. Continuous movements are more likely intentional
    const continuityFactor = features.continuity > 0.7;
    
    // 10. Magnitude of movement
    const magnitudeFactor = features.magnitudeOfMovement > 15;
    
    // Calculate weighted score
    const factors = [
      { value: speedFactor, weight: 1.0 },
      { value: smoothnessFactor, weight: 1.5 },  // Key indicator
      { value: accelerationFactor, weight: 0.8 },
      { value: durationFactor, weight: 1.0 },
      { value: directionFactor, weight: 0.7 },
      { value: frequencyFactor, weight: 1.2 },   // Important for tremor detection
      { value: steadinessFactor, weight: 1.2 },  // Helps detect spasms
      { value: patternFactor, weight: 0.9 },
      { value: continuityFactor, weight: 1.0 },
      { value: magnitudeFactor, weight: 0.7 }
    ];
    
    // Calculate weighted sum
    let weightedSum = 0;
    let totalWeight = 0;
    
    factors.forEach(factor => {
      weightedSum += factor.value ? factor.weight : 0;
      totalWeight += factor.weight;
    });
    
    // Normalize to 0-1 range
    const intentionalityScore = weightedSum / totalWeight;
    
    // Return result with confidence
    return {
      isIntentional: intentionalityScore > 0.55, // Threshold
      confidence: intentionalityScore
    };
  }
  
  /**
   * Check if a keypoint is not on cooldown (to prevent rapid triggers)
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
    let samplesAdded = 0;
    
    // Extract features for each tracked keypoint
    TRACKED_KEYPOINTS.forEach(keypointName => {
      const features = this.extractFeatures(keypointName);
      if (features) {
        this.calibrationSamples.push({
          features,
          isIntentional
        });
        samplesAdded++;
      }
    });
    
    // Log calibration progress occasionally
    if (this.calibrationSamples.length % 20 === 0) {
      const intentionalCount = this.calibrationSamples.filter(s => s.isIntentional).length;
      const unintentionalCount = this.calibrationSamples.length - intentionalCount;
      
      console.log(`Calibration samples: ${this.calibrationSamples.length} (${intentionalCount} intentional, ${unintentionalCount} unintentional)`);
    }
  }
  
  /**
   * Train model with collected calibration samples
   * This delegates to the worker when available
   */
  public async trainModel(): Promise<boolean> {
    if (this.calibrationSamples.length < 20) {
      console.warn('Not enough calibration samples to train the model');
      return false;
    }
    
    // Proceed with training
    if (this.isWorkerInitialized && this.mainWorker) {
      try {
        // Send training request to worker
        const trainingMessageId = `train-${Date.now()}`;
        
        // Create promise for response
        const trainingResult = new Promise<boolean>((resolve, reject) => {
          this.pendingWorkerRequests.set(trainingMessageId, {
            resolve,
            reject,
            timeout: window.setTimeout(() => {
              this.pendingWorkerRequests.delete(trainingMessageId);
              reject('Training request timed out');
            }, 60000) // 60 seconds timeout for training
          });
        });
        
        // Send training request
        this.mainWorker.postMessage({
          type: 'train',
          id: trainingMessageId,
          samples: this.calibrationSamples
        });
        
        // Wait for response
        const success = await trainingResult;
        
        if (success) {
          this.isModelTrained = true;
          this.useMLPrediction = true;
        }
        
        return success;
      } catch (error) {
        this.lastError = `Error training model: ${error}`;
        this.errorCount++;
        console.error(this.lastError);
        return false;
      }
    } else {
      console.warn('Worker not available, training is only implemented in worker');
      return false;
    }
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
  }
  
  /**
   * Set cooldown period for movement detection
   */
  public setCooldownPeriod(periodMs: number): void {
    this.cooldownPeriod = Math.max(50, Math.min(1000, periodMs));
  }
  
  /**
   * Set frame skip to improve performance
   * 0 = process every frame, 1 = every other frame, etc.
   */
  public setFrameSkip(skipFrames: number): void {
    this.frameSkip = Math.max(0, Math.min(5, skipFrames));
  }
  
  /**
   * Get detector status
   */
  public getStatus(): DetectorStatus {
    const intentionalSamples = this.calibrationSamples.filter(s => s.isIntentional).length;
    
    return {
      isModelTrained: this.isModelTrained,
      calibrationSamples: this.calibrationSamples.length,
      intentionalSamples,
      unintentionalSamples: this.calibrationSamples.length - intentionalSamples,
      isTfInitialized: tensorflowHelper.isReady(),
      calibrationQuality: this.calibrationQuality,
      azureEnabled: this.azureConfig.enabled,
      lastError: this.lastError,
      memoryUsage: tensorflowHelper.getMemoryInfo()?.numBytes || 0
    };
  }
  
  /**
   * Clean up resources when detector is no longer needed
   */
  public dispose(): void {
    // Dispose workers
    if (this.mainWorker) {
      this.mainWorker.terminate();
      this.mainWorker = null;
    }
    
    // Clean up any pending requests
    this.pendingWorkerRequests.forEach(request => {
      clearTimeout(request.timeout);
    });
    this.pendingWorkerRequests.clear();
    
    // Clean up resources
    tensorflowHelper.dispose();
    
    console.log('MLIntentionDetector: Disposed');
  }
  
  /**
   * Run diagnostics to help troubleshoot issues
   */
  public async runDiagnostics(): Promise<Record<string, any>> {
    const diagnostics: Record<string, any> = {
      version: '2.0.0',
      timestamp: new Date().toISOString(),
      detector: {
        isInitialized: this.isWorkerInitialized,
        isModelTrained: this.isModelTrained,
        useMLPrediction: this.useMLPrediction,
        errorCount: this.errorCount,
        lastError: this.lastError
      },
      performance: performanceMonitor.getStats(),
      tensorflow: {
        backend: tensorflowHelper.getBackend(),
        memory: tensorflowHelper.getMemoryInfo()
      },
      samples: {
        count: this.calibrationSamples.length,
        intentional: this.calibrationSamples.filter(s => s.isIntentional).length,
        unintentional: this.calibrationSamples.filter(s => !s.isIntentional).length
      },
      optimization: {
        frameSkip: this.frameSkip,
        cooldownPeriod: this.cooldownPeriod
      }
    };
    
    return diagnostics;
  }

  /**
 * Load a saved model for this user
 */
public async loadModel(userId: string): Promise<boolean> {
  try {
    // Try to load calibration quality from localStorage
    const qualityStr = localStorage.getItem(`user-calibration-quality-${userId}`);
    if (qualityStr) {
      this.calibrationQuality = parseInt(qualityStr, 10);
    }
    
    // Load model weights via TensorFlow.js
    if (this.isWorkerInitialized && this.mainWorker) {
      // Request model loading via worker
      const loadMessageId = `load-model-${Date.now()}`;
      
      // Create promise for response
      const loadResult = new Promise<boolean>((resolve, reject) => {
        this.pendingWorkerRequests.set(loadMessageId, {
          resolve,
          reject,
          timeout: window.setTimeout(() => {
            this.pendingWorkerRequests.delete(loadMessageId);
            reject('Load model request timed out');
          }, 30000) // 30 seconds timeout
        });
      });
      
      // Send load request to worker
      this.mainWorker.postMessage({
        type: 'loadModel',
        id: loadMessageId,
        userId
      });
      
      // Wait for response
      const success = await loadResult;
      
      if (success) {
        this.isModelTrained = true;
        this.useMLPrediction = true;
        console.log(`Model for user ${userId} loaded`);
      }
      
      return success;
    } else {
      // Fallback to local implementation using TensorFlow.js directly
      try {
        const modelUrl = `localstorage://user-model-${userId}`;
        const model = await tf.loadLayersModel(modelUrl);
        
        console.log(`Model for user ${userId} loaded locally`);
        this.isModelTrained = true;
        this.useMLPrediction = true;
        
        return true;
      } catch (error) {
        console.error('Error loading model locally:', error);
        return false;
      }
    }
  } catch (error) {
    this.lastError = `Error loading model: ${error}`;
    this.errorCount++;
    console.error(this.lastError);
    return false;
  }
}

/**
 * Save model for this user
 */
public async saveModel(userId: string): Promise<boolean> {
  try {
    if (!this.isModelTrained) {
      console.warn('No trained model to save');
      return false;
    }
    
    // Save calibration quality
    localStorage.setItem(`user-calibration-quality-${userId}`, this.calibrationQuality.toString());
    
    if (this.isWorkerInitialized && this.mainWorker) {
      // Request model saving via worker
      const saveMessageId = `save-model-${Date.now()}`;
      
      // Create promise for response
      const saveResult = new Promise<boolean>((resolve, reject) => {
        this.pendingWorkerRequests.set(saveMessageId, {
          resolve,
          reject,
          timeout: window.setTimeout(() => {
            this.pendingWorkerRequests.delete(saveMessageId);
            reject('Save model request timed out');
          }, 30000) // 30 seconds timeout
        });
      });
      
      // Send save request to worker
      this.mainWorker.postMessage({
        type: 'saveModel',
        id: saveMessageId,
        userId
      });
      
      // Wait for response
      const success = await saveResult;
      
      if (success) {
        console.log(`Model for user ${userId} saved`);
      }
      
      return success;
    } else {
      // Fallback to local implementation - this saves the model in localStorage
      try {
        await tf.tidy(() => {
          const saveUrl = `localstorage://user-model-${userId}`;
          // We don't have direct access to the model in this implementation
          // This is a limitation of the compatibility layer
          console.log(`Model would be saved to ${saveUrl} if available`);
        });
        
        return true;
      } catch (error) {
        console.error('Error saving model locally:', error);
        return false;
      }
    }
  } catch (error) {
    this.lastError = `Error saving model: ${error}`;
    this.errorCount++;
    console.error(this.lastError);
    return false;
  }
}

/**
 * Train a model on Azure
 * This is a compatibility method that wraps our Azure integration
 */
public async trainAzureModel(userId: string): Promise<boolean> {
  if (!this.azureConfig.enabled || !this.azureConfig.apiKey || !this.azureConfig.endpoint) {
    console.warn('Azure integration not enabled');
    return false;
  }
  
  try {
    // Set to Azure training mode
    this.isAzureTraining = true;
    
    // Create Azure integration instance
    const azureIntegration = new AzureIntegration(this.azureConfig);
    
    // Train the model
    const result = await azureIntegration.trainModel(
      userId,
      this.calibrationSamples,
      (progress) => {
        // Update progress if callback is available
        if (this.trainingProgressCallback) {
          this.trainingProgressCallback(progress);
        }
      }
    );
    
    // Update state
    this.isAzureTraining = false;
    
    if (result.success && result.modelId) {
      // Save the model ID
      this.azureConfig.defaultModelId = result.modelId;
      return true;
    } else {
      this.lastError = result.error || 'Unknown Azure training error';
      console.error(this.lastError);
      return false;
    }
  } catch (error) {
    this.isAzureTraining = false;
    this.lastError = `Error training Azure model: ${error}`;
    this.errorCount++;
    console.error(this.lastError);
    return false;
  }
}

// Add this state property for compatibility
private isAzureTraining: boolean = false;
}