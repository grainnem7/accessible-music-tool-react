import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs';
import { AzureCustomVisionService, IntentionalityPrediction } from '../services/azureCustomVisionService';
import { AzureMLService } from '../services/azureMLService';

// We're particularly interested in hand and upper body tracking
const TRACKED_KEYPOINTS = [
  'left_wrist', 'right_wrist',
  'left_elbow', 'right_elbow',
  'left_shoulder', 'right_shoulder',
  'left_index', 'right_index',  // Hand fingers - only available in BlazePose
  'left_thumb', 'right_thumb',  // Hand fingers - only available in BlazePose
  'left_pinky', 'right_pinky',  // Hand fingers - only available in BlazePose
  'nose', 'left_eye', 'right_eye'
];

// Interfaces for our data structures
export interface MovementFeatures {
  keypoint: string;
  velocityX: number;
  velocityY: number;
  acceleration: number;
  jitter: number;
  direction: string;
  isSmooth: boolean;
  timestamp: number;
  // Additional features for better classification
  magnitudeOfMovement: number;
  durationOfMovement: number;
  isReversing: boolean;
  // New features for better classification
  frequencyOfMovement: number;   // How often the direction changes
  steadiness: number;            // Measure of how steady a movement is
  patternScore: number;          // Score for pattern recognition
  continuity: number;            // Measure of continuous vs. interrupted movement
}

export interface MovementInfo {
  keypoint: string;
  isIntentional: boolean;
  velocity: number;
  direction: string;
  confidence: number;
}

export interface CalibrationSample {
  features: MovementFeatures;
  isIntentional: boolean;
}

interface AzureServiceConfig {
  apiKey: string;
  endpoint: string;
  enabled: boolean;
}

export class MLIntentionDetector {
  private posesHistory: Array<{poses: poseDetection.Pose[], timestamp: number}> = [];
  private historyLength = 60; // ~2 seconds at 30fps (increased for better pattern detection)
  private lastIntentionalMovements = new Map<string, number>(); // Map of keypoint to timestamp
  private cooldownPeriod = 200; // ms cooldown period to avoid rapid triggering
  private minConfidence = 0.5; // Minimum confidence for keypoints
  
  // Movement trajectory tracking
  private movementStart = new Map<string, {x: number, y: number, time: number}>();
  private isInMovement = new Map<string, boolean>();
  private movementPatterns = new Map<string, number[]>();// Store patterns for each keypoint

  // Machine learning model related properties
  private model: tf.LayersModel | null = null;
  private calibrationSamples: CalibrationSample[] = [];
  private isModelTrained = false;
  private useMLPrediction = false; // Only use ML after training
  private isTfInitialized = false;
  private trainingProgressCallback: ((progress: number) => void) | null = null;
  
  // Azure integration
  private azureConfig: AzureServiceConfig = {
    apiKey: '',
    endpoint: '',
    enabled: false
  };
  
  // Azure Custom Vision and ML services
  private azureCustomVisionService: AzureCustomVisionService;
  private azureMLService: AzureMLService;
  private useAzureCustomVision = false;
  private useAzureML = false;
  
  // Calibration extension
  private calibrationDuration = 15; // 15 seconds per calibration step (increased)
  private calibrationQuality = 0; // Score from 0-100 of calibration quality
  
  // Constructor
  constructor() {
    this.initTensorflow();
    
    // Initialize Azure services
    this.azureCustomVisionService = new AzureCustomVisionService();
    this.azureMLService = new AzureMLService();
  }
  
  // Initialize TensorFlow
  private async initTensorflow() {
    try {
      // Make sure TF is ready
      await tf.ready();
      
      // Set a stable backend
      await tf.setBackend('webgl');
      console.log('TensorFlow.js initialized with backend:', tf.getBackend());
      
      this.isTfInitialized = true;
      this.initModel();
    } catch (error) {
      console.error('Error initializing TensorFlow:', error);
    }
  }
  
  // Set up Azure integration (can be enabled/disabled)
  public configureAzureServices(config: AzureServiceConfig) {
    this.azureConfig = config;
    console.log('Azure Computer Vision integration ' + (config.enabled ? 'enabled' : 'disabled'));
  }
  
  // Configure Azure Custom Vision usage
  public setUseAzureCustomVision(use: boolean): void {
    this.useAzureCustomVision = use;
    console.log(`Azure Custom Vision ${use ? 'enabled' : 'disabled'}`);
  }
  
  // Configure Azure ML usage
  public setUseAzureML(use: boolean): void {
    this.useAzureML = use;
    console.log(`Azure ML ${use ? 'enabled' : 'disabled'}`);
  }
  
  // Add callback for training progress updates
  public setTrainingProgressCallback(callback: (progress: number) => void): void {
    this.trainingProgressCallback = callback;
  }
  
  // Create a more advanced neural network model
  private async initModel(): Promise<void> {
    if (!this.isTfInitialized) {
      console.warn('TensorFlow.js not initialized yet, waiting...');
      await this.waitForTensorflow();
    }
    
    try {
      const model = tf.sequential();
      
      // Input layer with expanded feature set
      model.add(tf.layers.dense({
        units: 64, // More units for a more sophisticated model
        activation: 'relu',
        inputShape: [15] // Increased number of features
      }));
      
      // Add dropout to prevent overfitting
      model.add(tf.layers.dropout({
        rate: 0.3
      }));
      
      // Additional hidden layers for more complex pattern recognition
      model.add(tf.layers.dense({
        units: 32,
        activation: 'relu'
      }));
      
      model.add(tf.layers.dropout({
        rate: 0.2
      }));
      
      model.add(tf.layers.dense({
        units: 16,
        activation: 'relu'
      }));
      
      // Output layer - binary classification
      model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
      }));
      
      // Compile model
      model.compile({
        optimizer: tf.train.adam(0.0005), // Lower learning rate for better generalization
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
      });
      
      this.model = model;
      console.log('Enhanced ML Intention Detection model initialized');
    } catch (error) {
      console.error('Error creating model:', error);
    }
  }
  
  // Helper method to wait for TensorFlow initialization
  private async waitForTensorflow(maxAttempts = 10): Promise<void> {
    let attempts = 0;
    
    while (!this.isTfInitialized && attempts < maxAttempts) {
      await new Promise(resolve => setTimeout(resolve, 500));
      attempts++;
    }
    
    if (!this.isTfInitialized) {
      throw new Error('TensorFlow initialization timed out');
    }
  }
  
  // Process new pose data
  public async processPoses(poses: poseDetection.Pose[]): Promise<MovementInfo[]> {
    // Add to history
    this.posesHistory.push({
      poses: poses,
      timestamp: Date.now()
    });
    
    // Trim history if it gets too long
    while (this.posesHistory.length > this.historyLength) {
      this.posesHistory.shift();
    }
    
    // Need at least a few frames to detect movement
    if (this.posesHistory.length < 10) {
      return [];
    }
    
    const movements: MovementInfo[] = [];
    
    // Only process if we have poses
    if (poses.length === 0) return movements;
    
    // Process each tracked keypoint
    for (const keypointName of TRACKED_KEYPOINTS) {
      const keypoint = this.findKeypointByName(poses[0], keypointName);
      
      if (keypoint && keypoint.score && keypoint.score > this.minConfidence) {
        // Extract features for this keypoint
        const features = this.extractFeatures(keypointName);
        
        if (!features) continue;
        
        let isIntentional = false;
        let confidence = 0.5;
        
        if (this.useAzureML) {
          // Use Azure ML for prediction
          try {
            const prediction = await this.azureMLService.predictIntentionality(features);
            isIntentional = prediction.isIntentional;
            confidence = prediction.confidence;
          } catch (err) {
            console.warn('Azure ML prediction failed, falling back to other methods', err);
            // Fall through to next method
          }
        } else if (this.useAzureCustomVision) {
          // Use Azure Custom Vision for prediction
          try {
            const prediction = await this.azureCustomVisionService.predictIntentionality(features);
            isIntentional = prediction.isIntentional;
            confidence = prediction.confidence;
          } catch (err) {
            console.warn('Azure Custom Vision prediction failed, falling back to local model', err);
            // Fall through to next method
          }
        } else if (this.isModelTrained && this.useMLPrediction) {
          // Use local model
          const prediction = this.predictWithModel(features);
          isIntentional = prediction.isIntentional;
          confidence = prediction.confidence;
        } else {
          // Fallback to improved heuristic approach
          isIntentional = this.heuristicIntentionality(features);
        }
        
        // Try Azure cognitive services if enabled and available
        if (this.azureConfig.enabled && features.magnitudeOfMovement > 10) {
          this.tryAzurePrediction(keypointName, features)
            .then(azurePrediction => {
              if (azurePrediction !== null) {
                // Blend the predictions
                isIntentional = azurePrediction.confidence > confidence 
                  ? azurePrediction.isIntentional 
                  : isIntentional;
                confidence = Math.max(confidence, azurePrediction.confidence);
              }
            })
            .catch(err => {
              console.warn('Azure prediction failed, using local prediction', err);
            });
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
            confidence: confidence
          });
        }
      }
    }
    
    return movements;
  }

  // Try to use Azure Computer Vision for prediction
  private async tryAzurePrediction(
    keypointName: string, 
    features: MovementFeatures
  ): Promise<{isIntentional: boolean, confidence: number} | null> {
    if (!this.azureConfig.enabled || !this.azureConfig.apiKey || !this.azureConfig.endpoint) {
      return null;
    }
    
    try {
      // Prepare data for Azure Custom Vision API
      const bodyData = {
        features: {
          velocityX: features.velocityX,
          velocityY: features.velocityY,
          acceleration: features.acceleration,
          jitter: features.jitter,
          isSmooth: features.isSmooth,
          direction: features.direction,
          magnitudeOfMovement: features.magnitudeOfMovement,
          frequencyOfMovement: features.frequencyOfMovement,
          steadiness: features.steadiness,
          patternScore: features.patternScore,
          continuity: features.continuity
        },
        keypoint: keypointName
      };

      // Make the actual API call to Azure
      const response = await fetch(`${this.azureConfig.endpoint}/intention/detect`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Ocp-Apim-Subscription-Key': this.azureConfig.apiKey
        },
        body: JSON.stringify(bodyData)
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Azure API error: ${response.status} - ${errorText}`);
      }

      const result = await response.json();
      
      // Extract the prediction result
      return {
        isIntentional: result.predictions.intentional > result.predictions.unintentional,
        confidence: Math.max(result.predictions.intentional, result.predictions.unintentional)
      };
    } catch (error) {
      console.error('Error calling Azure Computer Vision API:', error);
      return null;
    }
  }
  
  // Extract more comprehensive features from the pose history
  private extractFeatures(keypointName: string): MovementFeatures | null {
    if (this.posesHistory.length < 10) return null;
    
    // Use more of the history for better analysis
    const recentHistory = this.posesHistory.slice(-15);
    const oldestPose = recentHistory[0].poses[0];
    const newestPose = recentHistory[recentHistory.length - 1].poses[0];
    
    const oldKeypoint = this.findKeypointByName(oldestPose, keypointName);
    const newKeypoint = this.findKeypointByName(newestPose, keypointName);
    
    if (!oldKeypoint || !newKeypoint || oldKeypoint.x === undefined || newKeypoint.x === undefined) {
      return null;
    }
    
    // Calculate velocity
    const timeElapsed = recentHistory[recentHistory.length - 1].timestamp - recentHistory[0].timestamp;
    const velocityX = (newKeypoint.x - oldKeypoint.x) / (timeElapsed / 1000);
    const velocityY = (newKeypoint.y - oldKeypoint.y) / (timeElapsed / 1000);
    
    // Calculate magnitude of movement
    const dx = newKeypoint.x - oldKeypoint.x;
    const dy = newKeypoint.y - oldKeypoint.y;
    const magnitudeOfMovement = Math.sqrt(dx * dx + dy * dy);
    
    // Calculate acceleration
    let acceleration = 0;
    if (recentHistory.length >= 3) {
      const midIndex = Math.floor(recentHistory.length / 2);
      const midPose = recentHistory[midIndex].poses[0];
      const midKeypoint = this.findKeypointByName(midPose, keypointName);
      
      if (midKeypoint && midKeypoint.x !== undefined) {
        const midTime = recentHistory[midIndex].timestamp;
        
        const velX1 = (midKeypoint.x - oldKeypoint.x) / ((midTime - recentHistory[0].timestamp) / 1000);
        const velY1 = (midKeypoint.y - oldKeypoint.y) / ((midTime - recentHistory[0].timestamp) / 1000);
        
        const velX2 = (newKeypoint.x - midKeypoint.x) / ((recentHistory[recentHistory.length - 1].timestamp - midTime) / 1000);
        const velY2 = (newKeypoint.y - midKeypoint.y) / ((recentHistory[recentHistory.length - 1].timestamp - midTime) / 1000);
        
        const accX = (velX2 - velX1) / (timeElapsed / 1000);
        const accY = (velY2 - velY1) / (timeElapsed / 1000);
        
        acceleration = Math.sqrt(accX * accX + accY * accY);
      }
    }
    
    // Detect if a movement is starting
    const isMovementStarting = magnitudeOfMovement > 5 && !this.isInMovement.get(keypointName);
    
    // If movement is starting, record start position and time
    if (isMovementStarting) {
      this.movementStart.set(keypointName, {
        x: oldKeypoint.x,
        y: oldKeypoint.y,
        time: recentHistory[0].timestamp
      });
      this.isInMovement.set(keypointName, true);
    }
    
    // Detect if movement is ending
    const isMovementEnding = magnitudeOfMovement < 3 && this.isInMovement.get(keypointName);
    
    // If movement is ending, calculate duration
    let durationOfMovement = 0;
    let isReversing = false;
    
    if (isMovementEnding) {
      const start = this.movementStart.get(keypointName);
      if (start) {
        durationOfMovement = (recentHistory[recentHistory.length - 1].timestamp - start.time) / 1000;
        
        // Check if the movement reversed direction
        const totalDx = newKeypoint.x - start.x;
        const totalDy = newKeypoint.y - start.y;
        
        // If the movement ended close to where it started, it might be a reversing movement
        const totalDistance = Math.sqrt(totalDx * totalDx + totalDy * totalDy);
        if (totalDistance < magnitudeOfMovement * 0.5) {
          isReversing = true;
        }
      }
      
      this.isInMovement.set(keypointName, false);
    } else if (this.isInMovement.get(keypointName)) {
      // Movement is still ongoing
      const start = this.movementStart.get(keypointName);
      if (start) {
        durationOfMovement = (recentHistory[recentHistory.length - 1].timestamp - start.time) / 1000;
      }
    }
    
    // Calculate jitter (deviation from straight line)
    const jitter = this.calculateJitter(keypointName);
    
    // Calculate direction
    let direction = 'none';
    if (Math.abs(dx) > Math.abs(dy)) {
      direction = dx > 0 ? 'right' : 'left';
    } else {
      direction = dy > 0 ? 'down' : 'up';
    }
    
    // Determine if movement is smooth
    const isSmooth = jitter < 10; // Threshold for smoothness
    
    // Calculate frequency of direction changes
    const frequencyOfMovement = this.calculateFrequencyOfMovement(keypointName);
    
    // Calculate steadiness (inverse of jitter variance)
    const steadiness = this.calculateSteadiness(keypointName);
    
    // Calculate pattern recognition score
    const patternScore = this.calculatePatternScore(keypointName);
    
    // Calculate continuity
    const continuity = this.calculateContinuity(keypointName);
    
    return {
      keypoint: keypointName,
      velocityX,
      velocityY,
      acceleration,
      jitter,
      direction,
      isSmooth,
      timestamp: Date.now(),
      magnitudeOfMovement,
      durationOfMovement,
      isReversing,
      frequencyOfMovement,
      steadiness,
      patternScore,
      continuity
    };
  }
  
  // Calculate jitter as deviation from a straight line
  private calculateJitter(keypointName: string): number {
    if (this.posesHistory.length < 10) return 0;
    
    const recentHistory = this.posesHistory.slice(-15);
    
    // Extract x,y coordinates for this keypoint over recent frames
    const coordinates: {x: number, y: number, timestamp: number}[] = [];
    
    recentHistory.forEach(history => {
      const keypoint = this.findKeypointByName(history.poses[0], keypointName);
      if (keypoint && keypoint.x !== undefined && keypoint.y !== undefined) {
        coordinates.push({
          x: keypoint.x, 
          y: keypoint.y,
          timestamp: history.timestamp
        });
      }
    });
    
    if (coordinates.length < 5) return 0;
    
    // Calculate average movement vector
    const startPoint = coordinates[0];
    const endPoint = coordinates[coordinates.length - 1];
    const vectorX = endPoint.x - startPoint.x;
    const vectorY = endPoint.y - startPoint.y;
    const vectorLength = Math.sqrt(vectorX * vectorX + vectorY * vectorY);
    
    // If barely any movement, return 0
    if (vectorLength < 2) return 0;
    
    // Calculate average deviation from the straight line between start and end
    let totalDeviation = 0;
    
    for (let i = 1; i < coordinates.length - 1; i++) {
      const point = coordinates[i];
      
      // Calculate progress along the line (0 to 1)
      const timeProgress = (point.timestamp - startPoint.timestamp) / 
                          (endPoint.timestamp - startPoint.timestamp);
      
      // Expected position if movement was perfectly linear
      const expectedX = startPoint.x + (vectorX * timeProgress);
      const expectedY = startPoint.y + (vectorY * timeProgress);
      
      // Calculate deviation from expected position
      const deviationX = point.x - expectedX;
      const deviationY = point.y - expectedY;
      const deviation = Math.sqrt(deviationX * deviationX + deviationY * deviationY);
      
      totalDeviation += deviation;
    }
    
    return totalDeviation / (coordinates.length - 2); // Average deviation
  }
  
  // Calculate how frequently direction changes (tremors have high frequency)
  private calculateFrequencyOfMovement(keypointName: string): number {
    if (this.posesHistory.length < 10) return 0;
    
    const recentHistory = this.posesHistory.slice(-15);
    
    // Extract coordinates
    const coordinates: {x: number, y: number, timestamp: number}[] = [];
    
    recentHistory.forEach(history => {
      const keypoint = this.findKeypointByName(history.poses[0], keypointName);
      if (keypoint && keypoint.x !== undefined && keypoint.y !== undefined) {
        coordinates.push({
          x: keypoint.x, 
          y: keypoint.y,
          timestamp: history.timestamp
        });
      }
    });
    
    if (coordinates.length < 5) return 0;
    
    // Detect direction changes
    let directionChanges = 0;
    let prevDx = 0;
    let prevDy = 0;
    
    for (let i = 1; i < coordinates.length; i++) {
      const dx = coordinates[i].x - coordinates[i-1].x;
      const dy = coordinates[i].y - coordinates[i-1].y;
      
      // Check if X direction changed
      if ((dx > 0 && prevDx < 0) || (dx < 0 && prevDx > 0)) {
        directionChanges++;
      }
      
      // Check if Y direction changed
      if ((dy > 0 && prevDy < 0) || (dy < 0 && prevDy > 0)) {
        directionChanges++;
      }
      
      prevDx = dx;
      prevDy = dy;
    }
    
    // Calculate frequency (changes per second)
    const timeSpan = (coordinates[coordinates.length-1].timestamp - coordinates[0].timestamp) / 1000;
    return timeSpan > 0 ? directionChanges / timeSpan : 0;
  }
  
  // Train Azure Custom Vision model
  public async trainAzureModel(userId: string): Promise<boolean> {
    try {
      // First upload calibration samples to Custom Vision
      const uploadSuccess = await this.azureCustomVisionService.uploadTrainingData(this.calibrationSamples);
      
      if (!uploadSuccess) {
        console.error('Failed to upload calibration samples to Azure Custom Vision');
        return false;
      }
      
      // Then train the model
      const trainingSuccess = await this.azureCustomVisionService.trainModel();
      
      if (trainingSuccess) {
        // Save Azure model info for this user
        localStorage.setItem(`user-azure-model-${userId}`, 'true');
        this.useAzureCustomVision = true;
      }
      
      return trainingSuccess;
    } catch (error) {
      console.error('Error training Azure model:', error);
      return false;
    }
  }
  
  // Train Azure ML model
  public async trainAzureMLModel(userId: string): Promise<boolean> {
    try {
      const success = await this.azureMLService.trainModel(userId, this.calibrationSamples);
      
      if (success) {
        this.useAzureML = true;
        localStorage.setItem(`user-azure-ml-model-${userId}`, 'true');
      }
      
      return success;
    } catch (error) {
      console.error('Error training Azure ML model:', error);
      return false;
    }
  }

  // Calculate steadiness (low variance = steady, high = erratic)
  private calculateSteadiness(keypointName: string): number {
    if (this.posesHistory.length < 10) return 0;
    
    const recentHistory = this.posesHistory.slice(-15);
    
    // Extract velocities
    const velocities: number[] = [];
    
    for (let i = 1; i < recentHistory.length; i++) {
      const prevPose = recentHistory[i-1].poses[0];
      const currPose = recentHistory[i].poses[0];
      
      const prevKeypoint = this.findKeypointByName(prevPose, keypointName);
      const currKeypoint = this.findKeypointByName(currPose, keypointName);
      
      if (prevKeypoint && currKeypoint && 
          prevKeypoint.x !== undefined && prevKeypoint.y !== undefined &&
          currKeypoint.x !== undefined && currKeypoint.y !== undefined) {
        
        const dx = currKeypoint.x - prevKeypoint.x;
        const dy = currKeypoint.y - prevKeypoint.y;
        const velocity = Math.sqrt(dx*dx + dy*dy);
        
        velocities.push(velocity);
      }
    }
    
    if (velocities.length < 3) return 0;
    
    // Calculate variance of velocities
    const mean = velocities.reduce((sum, v) => sum + v, 0) / velocities.length;
    const variance = velocities.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / velocities.length;
    
    // Convert variance to steadiness (inverse relationship)
    // Higher variance means less steady
    return Math.max(0, 1 - Math.min(1, variance / 100));
  }
  
  // Calculate pattern matching score
  private calculatePatternScore(keypointName: string): number {
    if (this.posesHistory.length < 10) return 0;
    
    // Extract movement pattern
    const recentHistory = this.posesHistory.slice(-15);
    const pattern: number[][] = [];
    
    for (let i = 1; i < recentHistory.length; i++) {
      const prevPose = recentHistory[i-1].poses[0];
      const currPose = recentHistory[i].poses[0];
      
      const prevKeypoint = this.findKeypointByName(prevPose, keypointName);
      const currKeypoint = this.findKeypointByName(currPose, keypointName);
      
      if (prevKeypoint && currKeypoint && 
          prevKeypoint.x !== undefined && prevKeypoint.y !== undefined &&
          currKeypoint.x !== undefined && currKeypoint.y !== undefined) {
        
        const dx = currKeypoint.x - prevKeypoint.x;
        const dy = currKeypoint.y - prevKeypoint.y;
        
        pattern.push([dx, dy]);
      }
    }
    
    if (pattern.length < 5) return 0;
    
    this.movementPatterns.set(keypointName, pattern.reduce((acc, val) => acc.concat(val), []));
    
    // For intentional movements, patterns often have a structure
    // We'll look for repeating patterns or consistent directions
    
    // Check for consistent direction
    let consistentXDirection = true;
    let consistentYDirection = true;
    const firstX = pattern[0][0];
    const firstY = pattern[0][1];
    
    for (let i = 1; i < pattern.length; i++) {
      if ((pattern[i][0] > 0) !== (firstX > 0)) consistentXDirection = false;
      if ((pattern[i][1] > 0) !== (firstY > 0)) consistentYDirection = false;
    }
    
    // Higher score if direction is consistent
    return (consistentXDirection || consistentYDirection) ? 0.8 : 0.2;
  }
  
  // Calculate continuity of movement
  private calculateContinuity(keypointName: string): number {
    if (this.posesHistory.length < 10) return 0;
    
    const recentHistory = this.posesHistory.slice(-15);
    
 // Extract velocities at each timestep
 const velocities: number[] = [];
    
 for (let i = 1; i < recentHistory.length; i++) {
   const prevPose = recentHistory[i-1].poses[0];
   const currPose = recentHistory[i].poses[0];
   const timeDiff = (recentHistory[i].timestamp - recentHistory[i-1].timestamp) / 1000;
   
   const prevKeypoint = this.findKeypointByName(prevPose, keypointName);
   const currKeypoint = this.findKeypointByName(currPose, keypointName);
   
   if (prevKeypoint && currKeypoint && 
       prevKeypoint.x !== undefined && prevKeypoint.y !== undefined &&
       currKeypoint.x !== undefined && currKeypoint.y !== undefined &&
       timeDiff > 0) {
     
     const dx = currKeypoint.x - prevKeypoint.x;
     const dy = currKeypoint.y - prevKeypoint.y;
     const velocity = Math.sqrt(dx*dx + dy*dy) / timeDiff;
     
     velocities.push(velocity);
   }
 }
 
 if (velocities.length < 3) return 0;
 
 // Count how many velocities are close to zero (pauses in movement)
 const pauseThreshold = 5; // Velocity below this is considered a pause
 const pauses = velocities.filter(v => v < pauseThreshold).length;
 
 // Calculate continuity as percentage of non-paused frames
 return Math.max(0, 1 - (pauses / velocities.length));
}

// Improved heuristic approach to determine intentionality
private heuristicIntentionality(features: MovementFeatures): boolean {
 const velocity = Math.sqrt(
   features.velocityX * features.velocityX + 
   features.velocityY * features.velocityY
 );
 
 // Enhanced approach using all our new features:
 
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
 
 // Calculate weighted score - intentional movements typically exhibit
 // more of these characteristics
 const factors = [
   { value: speedFactor, weight: 1.0 },
   { value: smoothnessFactor, weight: 1.5 },  // Smoothness is a key indicator
   { value: accelerationFactor, weight: 0.8 },
   { value: durationFactor, weight: 1.0 },
   { value: directionFactor, weight: 0.7 },
   { value: frequencyFactor, weight: 1.2 },   // Frequency is important for tremor detection
   { value: steadinessFactor, weight: 1.2 },  // Steadiness helps detect spasms
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
 
 // Debugging
 console.log(`Intentionality for ${features.keypoint}: ${intentionalityScore.toFixed(2)}`);
 console.log(`  Speed: ${speedFactor}, Smooth: ${smoothnessFactor}, Accel: ${accelerationFactor}`);
 console.log(`  Duration: ${durationFactor}, Direction: ${directionFactor}, Frequency: ${frequencyFactor}`);
 console.log(`  Steadiness: ${steadinessFactor}, Pattern: ${patternFactor}, Continuity: ${continuityFactor}`);
 
 // Use threshold that can be tuned
 return intentionalityScore > 0.55; // Adjusted threshold
}

// Check cooldown to prevent rapid triggers
private isNotOnCooldown(keypointName: string): boolean {
 const lastMovement = this.lastIntentionalMovements.get(keypointName);
 if (!lastMovement) return true;
 
 const now = Date.now();
 return (now - lastMovement) > this.cooldownPeriod;
}

// Helper to find a keypoint by name
private findKeypointByName(pose: poseDetection.Pose, name: string): poseDetection.Keypoint | undefined {
 return pose.keypoints.find(kp => kp.name === name);
}

// Add calibration sample
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
   } else {
     console.warn(`Could not extract features for keypoint: ${keypointName}`);
   }
 });
 
 if (samplesAdded === 0) {
   console.warn('No features could be extracted for any keypoints in this frame');
 } else {
   console.log(`Added ${samplesAdded} calibration samples (${isIntentional ? 'intentional' : 'unintentional'})`);
 }
 
 const intentionalCount = this.calibrationSamples.filter(s => s.isIntentional).length;
 const unintentionalCount = this.calibrationSamples.filter(s => !s.isIntentional).length;
 
 console.log(`Total calibration samples: ${this.calibrationSamples.length} (${intentionalCount} intentional, ${unintentionalCount} unintentional)`);
}

// Train the model with collected samples
public async trainModel(): Promise<boolean> {
 if (!this.isTfInitialized) {
   console.warn('TensorFlow.js not initialized yet, waiting...');
   try {
     await this.waitForTensorflow();
   } catch (error) {
     console.error('TensorFlow initialization failed:', error);
     return false;
   }
 }
 
 if (!this.model) {
   console.error('ML model not initialized');
   return false;
 }
 
 if (this.calibrationSamples.length < 10) {
   console.warn('Not enough calibration samples to train the model');
   return false;
 }
 
 // Calculate calibration quality score
 await this.calculateCalibrationQuality();
 
 if (this.calibrationQuality < 50) {
   console.warn('Low calibration quality. Consider recalibrating with more distinct movements.');
   // Still proceed with training, but warn the user
 }
 
 // Prepare training data with enhanced feature set
 const featureValues = this.calibrationSamples.map(sample => [
   sample.features.velocityX,
   sample.features.velocityY,
   sample.features.acceleration,
   sample.features.jitter,
   sample.features.isSmooth ? 1 : 0,
   // Direction encoded
   sample.features.direction === 'up' ? 1 : 0,
   sample.features.direction === 'down' ? 1 : 0,
   sample.features.direction === 'left' ? 1 : 0,
   sample.features.direction === 'right' ? 1 : 0,
   // Additional features
   sample.features.magnitudeOfMovement / 100, // Normalize
   sample.features.isReversing ? 1 : 0,
   // New features
   sample.features.frequencyOfMovement / 10, // Normalize
   sample.features.steadiness,
   sample.features.patternScore,
   sample.features.continuity
 ]);
 
 const labels = this.calibrationSamples.map(sample => 
   sample.isIntentional ? 1 : 0
 );
 
 // Add data augmentation and balancing
 const augmentedFeatures = [...featureValues];
 const augmentedLabels = [...labels];
 
 // Count samples in each class
 const intentionalCount = labels.filter(l => l === 1).length;
 const unintentionalCount = labels.filter(l => l === 0).length;
 
 // Balance the dataset by augmenting the minority class
 if (intentionalCount < unintentionalCount) {
   // Augment intentional samples
   for (let i = 0; i < featureValues.length; i++) {
     if (labels[i] === 1) {
       // Create variations of this sample
       for (let j = 0; j < Math.min(5, Math.floor(unintentionalCount / intentionalCount)); j++) {
         const newSample = [...featureValues[i]];
         // Add slight random variations to numerical features
         newSample[0] *= 0.9 + (Math.random() * 0.2); // velocityX
         newSample[1] *= 0.9 + (Math.random() * 0.2); // velocityY
         newSample[2] *= 0.9 + (Math.random() * 0.2); // acceleration
         newSample[3] *= 0.9 + (Math.random() * 0.2); // jitter
         newSample[9] *= 0.9 + (Math.random() * 0.2); // magnitudeOfMovement
         
         augmentedFeatures.push(newSample);
         augmentedLabels.push(1);
       }
     }
   }
 } else if (unintentionalCount < intentionalCount) {
   // Augment unintentional samples
   for (let i = 0; i < featureValues.length; i++) {
     if (labels[i] === 0) {
       // Create variations of this sample
       for (let j = 0; j < Math.min(5, Math.floor(intentionalCount / unintentionalCount)); j++) {
         const newSample = [...featureValues[i]];
         // Add slight random variations
         newSample[0] *= 0.9 + (Math.random() * 0.2);
         newSample[1] *= 0.9 + (Math.random() * 0.2);
         newSample[2] *= 0.9 + (Math.random() * 0.2);
         newSample[3] *= 0.9 + (Math.random() * 0.2);
         newSample[9] *= 0.9 + (Math.random() * 0.2);
         
         augmentedFeatures.push(newSample);
         augmentedLabels.push(0);
       }
     }
   }
 }
 console.log(`Augmented dataset: ${augmentedFeatures.length} samples (original: ${featureValues.length})`);
 
 // Create tensors
 const xs = tf.tensor2d(augmentedFeatures);
 const ys = tf.tensor2d(augmentedLabels, [augmentedLabels.length, 1]);
 
 // Train the model
 try {
   console.log('Training model with', augmentedFeatures.length, 'samples');
   
   const totalEpochs = 100; // Increased for better training
   
   const trainingConfig = {
     epochs: totalEpochs,
     batchSize: 32,
     validationSplit: 0.2, // Hold out 20% of data for validation
     callbacks: {
       onEpochEnd: (epoch: number, logs: any) => {
         console.log(`Epoch ${epoch}: loss = ${logs.loss}, accuracy = ${logs.acc}, val_loss = ${logs.val_loss}, val_acc = ${logs.val_acc}`);
         
         // Report progress
         if (this.trainingProgressCallback) {
           const progress = (epoch + 1) / totalEpochs;
           this.trainingProgressCallback(progress);
         }
       }
     }
   };
   
   await this.model.fit(xs, ys, trainingConfig);
   
   // Check model accuracy
   const result = this.model.evaluate(xs, ys) as tf.Tensor[];
   const accuracy = result[1].dataSync()[0];
   
   console.log(`Model training complete. Final accuracy: ${accuracy}`);
   
   this.isModelTrained = true;
   this.useMLPrediction = true;
   
   // Clean up tensors
   xs.dispose();
   ys.dispose();
   result.forEach(t => t.dispose());
   
   // Final progress update
   if (this.trainingProgressCallback) {
     this.trainingProgressCallback(1.0);
   }
   
   return true;
 } catch (error) {
   console.error('Error training model:', error);
   
   // Clean up tensors on error
   xs.dispose();
   ys.dispose();
   
   return false;
 }
}

// Calculate calibration quality score
private async calculateCalibrationQuality(): Promise<void> {
 // Factors affecting calibration quality:
 // 1. Number of samples
 // 2. Balance between intentional and unintentional samples
 // 3. Diversity of movements
 // 4. Clear distinction between intentional and unintentional patterns
 
 // Count samples
 const intentionalSamples = this.calibrationSamples.filter(s => s.isIntentional).length;
 const unintentionalSamples = this.calibrationSamples.filter(s => !s.isIntentional).length;
 const totalSamples = this.calibrationSamples.length;
 
 // Calculate sample count score (0-25)
 const minRequiredSamples = 30;
 const idealSamples = 100;
 const sampleCountScore = Math.min(25, Math.round(25 * (totalSamples / idealSamples)));
 
 // Calculate balance score (0-25)
 const maxImbalanceRatio = 0.7; // Ideal is 0.5 (perfect balance)
 const balance = Math.min(intentionalSamples, unintentionalSamples) / 
                 Math.max(intentionalSamples, unintentionalSamples);
 const balanceScore = Math.min(25, Math.round(25 * (balance / maxImbalanceRatio)));
 
 // Calculate movement diversity score (0-25)
 // Look at the variety of movements in each category
 const diversityScore = this.calculateMovementDiversity();
 
 // Calculate distinction score (0-25)
 // How well can we separate intentional from unintentional movements?
 const distinctionScore = await this.calculateIntentionalDistinction();
 
 // Overall quality score (0-100)
 this.calibrationQuality = sampleCountScore + balanceScore + diversityScore + distinctionScore;
 
 console.log(`Calibration quality: ${this.calibrationQuality}/100`);
 console.log(`- Sample count: ${sampleCountScore}/25 (${totalSamples} samples)`);
 console.log(`- Balance: ${balanceScore}/25 (${intentionalSamples} intentional, ${unintentionalSamples} unintentional)`);
 console.log(`- Diversity: ${diversityScore}/25`);
 console.log(`- Distinction: ${distinctionScore}/25`);
}

// Calculate movement diversity score
private calculateMovementDiversity(): number {
 // Analyze the variety of movement patterns in the calibration samples
 
 const intentionalPatterns = this.calibrationSamples
   .filter(s => s.isIntentional)
   .map(s => [s.features.direction, Math.round(s.features.magnitudeOfMovement / 10)]);
 
 const unintentionalPatterns = this.calibrationSamples
   .filter(s => !s.isIntentional)
   .map(s => [s.features.direction, Math.round(s.features.magnitudeOfMovement / 10)]);
 
 // Count unique patterns
 const uniqueIntentionalPatterns = new Set(intentionalPatterns.map(p => p.join('-'))).size;
 const uniqueUnintentionalPatterns = new Set(unintentionalPatterns.map(p => p.join('-'))).size;
 
 // Calculate diversity score
 const diversityScore = Math.min(25, Math.round(
   12.5 * (uniqueIntentionalPatterns / Math.max(5, intentionalPatterns.length / 3)) +
   12.5 * (uniqueUnintentionalPatterns / Math.max(5, unintentionalPatterns.length / 3))
 ));
 
 return diversityScore;
}

// Calculate how well intentional and unintentional movements can be distinguished
private async calculateIntentionalDistinction(): Promise<number> {
 // Simple approach: Train a small model on a subset of the data and test
 // on the remaining data. The accuracy is the distinction score.
 
 if (this.calibrationSamples.length < 20) {
   return 10; // Not enough data for a good test
 }
 
 try {
   // Extract features that are good indicators
   const features = this.calibrationSamples.map(sample => [
     sample.features.jitter,
     sample.features.isSmooth ? 1 : 0,
     sample.features.magnitudeOfMovement / 100,
     sample.features.isReversing ? 1 : 0
   ]);
   
   const labels = this.calibrationSamples.map(sample => 
     sample.isIntentional ? 1 : 0
   );
   
   // Split into training and testing sets (70/30)
   const splitIndex = Math.floor(features.length * 0.7);
   const trainFeatures = features.slice(0, splitIndex);
   const testFeatures = features.slice(splitIndex);
   const trainLabels = labels.slice(0, splitIndex);
   const testLabels = labels.slice(splitIndex);
   
   // Check if we have both classes in the test set
   const hasPositive = testLabels.some(l => l === 1);
   const hasNegative = testLabels.some(l => l === 0);
   
   if (!hasPositive || !hasNegative) {
     return 15; // Not enough diversity
   }
   
   // Create tensors
   const xsTrain = tf.tensor2d(trainFeatures);
   const ysTrain = tf.tensor2d(trainLabels, [trainLabels.length, 1]);
   const xsTest = tf.tensor2d(testFeatures);
   const ysTest = tf.tensor2d(testLabels, [testLabels.length, 1]);
   
   // Create a simple model
   const model = tf.sequential();
   model.add(tf.layers.dense({
     units: 8,
     activation: 'relu',
     inputShape: [4]
   }));
   model.add(tf.layers.dense({
     units: 1,
     activation: 'sigmoid'
   }));
   
   model.compile({
     optimizer: tf.train.adam(0.01),
     loss: 'binaryCrossentropy',
     metrics: ['accuracy']
   });
   
   // Train silently (no callbacks)
   await model.fit(xsTrain, ysTrain, {
     epochs: 20,
     batchSize: 16,
     verbose: 0
   });
   
   // Evaluate
   const result = model.evaluate(xsTest, ysTest) as tf.Tensor[];
   const accuracy = result[1].dataSync()[0];
   
   // Clean up
   xsTrain.dispose();
   ysTrain.dispose();
   xsTest.dispose();
   ysTest.dispose();
   result.forEach(t => t.dispose());
   model.dispose();
   
   // Calculate distinction score (0-25)
   // 0.5 accuracy is random, 1.0 is perfect
   const distinctionScore = Math.round(25 * (accuracy - 0.5) * 2);
   return Math.max(0, Math.min(25, distinctionScore));
   
 } catch (error) {
   console.error('Error calculating distinction score:', error);
   return 10; // Default value on error
 }
}

// Make a prediction with the trained model
private predictWithModel(features: MovementFeatures): {isIntentional: boolean, confidence: number} {
 if (!this.model || !this.isModelTrained || !this.isTfInitialized) {
   return { isIntentional: false, confidence: 0.5 };
 }
 
 // Prepare feature vector with all features
 const featureArray = [
   features.velocityX,
   features.velocityY,
   features.acceleration,
   features.jitter,
   features.isSmooth ? 1 : 0,
   // Direction encoded
   features.direction === 'up' ? 1 : 0,
   features.direction === 'down' ? 1 : 0,
   features.direction === 'left' ? 1 : 0,
   features.direction === 'right' ? 1 : 0,
   // Additional features
   features.magnitudeOfMovement / 100, // Normalize
   features.isReversing ? 1 : 0,
   // New features
   features.frequencyOfMovement / 10, // Normalize
   features.steadiness,
   features.patternScore,
   features.continuity
 ];
 
 // Make prediction
 const input = tf.tensor2d([featureArray]);
 
 try {
   const prediction = this.model.predict(input) as tf.Tensor;
   const value = prediction.dataSync()[0];
   
   // Clean up tensors
   input.dispose();
   prediction.dispose();
   
   // Return prediction with confidence
   return { 
     isIntentional: value > 0.65, // Threshold for classification
     confidence: value          // Raw confidence score
   };
 } catch (error) {
   console.error('Prediction error:', error);
   input.dispose();
   return { isIntentional: false, confidence: 0.5 };
 }
}

// Save the model for this user
public async saveModel(userId: string): Promise<boolean> {
 if (!this.model || !this.isModelTrained || !this.isTfInitialized) {
   console.error('No trained model to save');
   return false;
 }
 
 try {
   // Save to local storage
   const saveResults = await this.model.save(`localstorage://user-model-${userId}`);
   console.log('Model saved:', saveResults);
   
   // Also save calibration quality
   localStorage.setItem(`user-calibration-quality-${userId}`, this.calibrationQuality.toString());
   
   return true;
 } catch (error) {
   console.error('Error saving model:', error);
   return false;
 }
}

// Load a saved model for this user
public async loadModel(userId: string): Promise<boolean> {
 if (!this.isTfInitialized) {
   console.warn('TensorFlow.js not initialized yet, waiting...');
   try {
     await this.waitForTensorflow();
   } catch (error) {
     console.error('TensorFlow initialization failed:', error);
     return false;
   }
 }
 
 try {
   // Try to load calibration quality
   const qualityStr = localStorage.getItem(`user-calibration-quality-${userId}`);
   if (qualityStr) {
     this.calibrationQuality = parseInt(qualityStr, 10);
   }
   
   const model = await tf.loadLayersModel(`localstorage://user-model-${userId}`);
   
   if (model) {
     // Replace current model
     if (this.model) {
       this.model.dispose();
     }
     
     this.model = model;
     this.isModelTrained = true;
     this.useMLPrediction = true;
     
     console.log('Loaded saved model for user', userId);
     console.log(`Calibration quality: ${this.calibrationQuality}/100`);
     return true;
   }
   
   return false;
 } catch (error) {
   console.error('Error loading model:', error);
   return false;
 }
}

// Clear calibration samples
public clearCalibration(): void {
 this.calibrationSamples = [];
 this.calibrationQuality = 0;
 console.log('Calibration samples cleared');
}

// Set calibration duration
public setCalibrationDuration(seconds: number): void {
 this.calibrationDuration = Math.max(5, seconds);
 console.log(`Calibration duration set to ${this.calibrationDuration} seconds`);
}

// Get calibration duration
public getCalibrationDuration(): number {
 return this.calibrationDuration;
}

// Get the status of the detector
public getStatus(): {
 isModelTrained: boolean;
 calibrationSamples: number;
 intentionalSamples: number;
 unintentionalSamples: number;
 isTfInitialized: boolean;
 calibrationQuality: number;
 azureEnabled: boolean;
} {
 const intentionalSamples = this.calibrationSamples.filter(s => s.isIntentional).length;
 
 return {
   isModelTrained: this.isModelTrained,
   calibrationSamples: this.calibrationSamples.length,
   intentionalSamples,
   unintentionalSamples: this.calibrationSamples.length - intentionalSamples,
   isTfInitialized: this.isTfInitialized,
   calibrationQuality: this.calibrationQuality,
   azureEnabled: this.azureConfig.enabled
 };
}
}