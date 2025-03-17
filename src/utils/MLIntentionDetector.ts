import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs';

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

export class MLIntentionDetector {
  private posesHistory: Array<{poses: poseDetection.Pose[], timestamp: number}> = [];
  private historyLength = 30; // ~1 second at 30fps
  private lastIntentionalMovements = new Map<string, number>(); // Map of keypoint to timestamp
  private cooldownPeriod = 200; // ms cooldown period to avoid rapid triggering (reduced for better responsiveness)
  private minConfidence = 0.5; // Reduced minimum confidence for more detection coverage
  
  // Movement trajectory tracking
  private movementStart = new Map<string, {x: number, y: number, time: number}>();
  private isInMovement = new Map<string, boolean>();

  // Machine learning model related properties
  private model: tf.LayersModel | null = null;
  private calibrationSamples: CalibrationSample[] = [];
  private isModelTrained = false;
  private useMLPrediction = false; // Only use ML after training
  private isTfInitialized = false;
  private trainingProgressCallback: ((progress: number) => void) | null = null;
  
  // Constructor
  constructor() {
    this.initTensorflow();
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
      
      // Input layer - we'll have more features now
      model.add(tf.layers.dense({
        units: 32, // Increased units for more complex patterns
        activation: 'relu',
        inputShape: [11] // Increased number of features
      }));
      
      // Add dropout to prevent overfitting
      model.add(tf.layers.dropout({
        rate: 0.2
      }));
      
      // Additional hidden layer for more complex pattern recognition
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
        optimizer: tf.train.adam(0.001), // Explicit learning rate
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
  public processPoses(poses: poseDetection.Pose[]): MovementInfo[] {
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
    if (this.posesHistory.length < 5) {
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
        
        // If we have a trained model, use it
        if (this.isModelTrained && this.useMLPrediction) {
          isIntentional = this.predictWithModel(features);
        } else {
          // Fallback to heuristic approach
          isIntentional = this.heuristicIntentionality(features);
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
            confidence: keypoint.score
          });
        }
      }
    }
    
    return movements;
  }
  
  // Extract more comprehensive features from the pose history
  private extractFeatures(keypointName: string): MovementFeatures | null {
    if (this.posesHistory.length < 5) return null;
    
    const recentHistory = this.posesHistory.slice(-5);
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
      isReversing
    };
  }
  
  // Calculate jitter as deviation from a straight line
  private calculateJitter(keypointName: string): number {
    if (this.posesHistory.length < 5) return 0;
    
    const recentHistory = this.posesHistory.slice(-5);
    
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
    
    if (coordinates.length < 3) return 0;
    
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
  
  // Improved heuristic approach to determine intentionality
  private heuristicIntentionality(features: MovementFeatures): boolean {
    const velocity = Math.sqrt(
      features.velocityX * features.velocityX + 
      features.velocityY * features.velocityY
    );
    
    // A more sophisticated approach combining multiple factors:
    
    // 1. Fast, deliberate movements are likely intentional
    const speedFactor = velocity > 25;
    
    // 2. Smooth movements with low jitter are likely intentional
    const smoothnessFactor = features.jitter < 12 && features.isSmooth;
    
    // 3. Movements with higher acceleration are likely intentional
    const accelerationFactor = features.acceleration > 15;
    
    // 4. Movements with a specific duration range are likely intentional
    const durationFactor = features.durationOfMovement > 0.2 && features.durationOfMovement < 1.5;
    
    // 5. Non-reversing movements are more likely intentional
    const directionFactor = !features.isReversing;
    
    // Combine factors - need at least 3 of the 5 factors to be true
    let intentionalFactorCount = 0;
    if (speedFactor) intentionalFactorCount++;
    if (smoothnessFactor) intentionalFactorCount++;
    if (accelerationFactor) intentionalFactorCount++;
    if (durationFactor) intentionalFactorCount++;
    if (directionFactor) intentionalFactorCount++;
    
    return intentionalFactorCount >= 3;
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
    // Extract features for each tracked keypoint
    TRACKED_KEYPOINTS.forEach(keypointName => {
      const features = this.extractFeatures(keypointName);
      if (features) {
        this.calibrationSamples.push({
          features,
          isIntentional
        });
      }
    });
    
    console.log(`Added calibration samples (${isIntentional ? 'intentional' : 'unintentional'})`);
    console.log(`Total calibration samples: ${this.calibrationSamples.length}`);
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
    
    // Prepare training data
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
      sample.features.isReversing ? 1 : 0
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
          for (let j = 0; j < Math.min(3, Math.floor(unintentionalCount / intentionalCount)); j++) {
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
          for (let j = 0; j < Math.min(3, Math.floor(intentionalCount / unintentionalCount)); j++) {
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
      
      const totalEpochs = 50;
      
      const trainingConfig = {
        epochs: totalEpochs,
        batchSize: 32,
        callbacks: {
          onEpochEnd: (epoch: number, logs: any) => {
            console.log(`Epoch ${epoch}: loss = ${logs.loss}, accuracy = ${logs.acc}`);
            
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
  
  // Make a prediction with the trained model
  private predictWithModel(features: MovementFeatures): boolean {
    if (!this.model || !this.isModelTrained || !this.isTfInitialized) {
      return false;
    }
    
    // Prepare feature vector
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
      features.isReversing ? 1 : 0
    ];
    
    // Make prediction
    const input = tf.tensor2d([featureArray]);
    
    try {
      const prediction = this.model.predict(input) as tf.Tensor;
      const value = prediction.dataSync()[0];
      
      // Clean up tensors
      input.dispose();
      prediction.dispose();
      
      // Return true if prediction confidence is above threshold
      return value > 0.65; // Slightly lowered threshold for better recall
    } catch (error) {
      console.error('Prediction error:', error);
      input.dispose();
      return false;
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
    console.log('Calibration samples cleared');
  }
  
  // Get the status of the detector
  public getStatus(): {
    isModelTrained: boolean;
    calibrationSamples: number;
    intentionalSamples: number;
    unintentionalSamples: number;
    isTfInitialized: boolean;
  } {
    const intentionalSamples = this.calibrationSamples.filter(s => s.isIntentional).length;
    
    return {
      isModelTrained: this.isModelTrained,
      calibrationSamples: this.calibrationSamples.length,
      intentionalSamples,
      unintentionalSamples: this.calibrationSamples.length - intentionalSamples,
      isTfInitialized: this.isTfInitialized
    };
  }
}