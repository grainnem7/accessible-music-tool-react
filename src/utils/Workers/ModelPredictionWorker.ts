import * as tf from '@tensorflow/tfjs';
import { InitMessage, PredictMessage, CleanupMessage, PredictionResultMessage, MovementFeatures } from '../IntentionDetection/DetectionTypes';

/**
 * Web Worker for TensorFlow-based prediction
 * Performs model inference in a background thread
 */

// TensorFlow model
let model: tf.LayersModel | null = null;
let isInitialized = false;

// Keep track of startup time for performance monitoring
const startTime = Date.now();

// Ensure the worker is registered properly
console.log('ModelPredictionWorker: Starting initialization');

// Listen for messages from main thread
self.addEventListener('message', async (event: MessageEvent) => {
  const message = event.data;
  
  console.log('ModelPredictionWorker: Received message', message.type);
  
  // Handle different message types
  switch (message.type) {
    case 'init':
      await handleInit(message);
      break;
      
    case 'predict':
      handlePredict(message);
      break;
      
    case 'cleanup':
      handleCleanup(message);
      break;
      
    default:
      console.error('ModelPredictionWorker: Unknown message type', message.type);
  }
});

/**
 * Initialize TensorFlow and load model
 */
async function handleInit(message: any): Promise<void> {
  try {
    console.log('ModelPredictionWorker: Initializing TensorFlow...');
    
    // Initialize TensorFlow
    await tf.ready();
    
    // Set preferred backend
    if (message.preferredBackend) {
      try {
        await tf.setBackend(message.preferredBackend);
        console.log(`ModelPredictionWorker: Using ${tf.getBackend()} backend`);
      } catch (error) {
        console.warn(`ModelPredictionWorker: Failed to set ${message.preferredBackend} backend, using default`);
      }
    }
    
    // Create the model if it doesn't exist
    if (!model) {
      createModel();
    }
    
    // If model weights were provided, load them
    if (message.modelWeights) {
      try {
        // Simple approach - directly load weights from array buffer
        const tensors = message.weightSpecs?.map((spec: any, i: number) => {
          // Create tensors from specs
          return tf.tensor(
            new Float32Array(message.modelWeights.slice(
              spec.offset || 0, 
              (spec.offset || 0) + (spec.size || 0) * 4
            )),
            spec.shape,
            spec.dtype
          );
        }) || [];
        
        if (model && tensors.length > 0) {
          model.setWeights(tensors);
          console.log('ModelPredictionWorker: Applied model weights');
        }
      } catch (error) {
        console.error('ModelPredictionWorker: Error loading model weights', error);
      }
    }
    
    isInitialized = true;
    
    // Send success response
    self.postMessage({
      type: 'initialized',
      id: message.id,
      backend: tf.getBackend(),
      initTimeMs: Date.now() - startTime
    });
  } catch (error) {
    console.error('ModelPredictionWorker: Error initializing', error);
    
    // Send error response
    self.postMessage({
      type: 'error',
      error: String(error),
      id: message.id
    });
  }
}

/**
 * Create a new model for intention detection
 */
function createModel(): void {
  try {
    // Create sequential model
    const sequential = tf.sequential();
    
    // Input layer with expanded feature set
    sequential.add(tf.layers.dense({
      units: 64,
      activation: 'relu',
      inputShape: [15] // Number of features
    }));
    
    // Dropout for regularization
    sequential.add(tf.layers.dropout({
      rate: 0.3
    }));
    
    // Hidden layers
    sequential.add(tf.layers.dense({
      units: 32,
      activation: 'relu'
    }));
    
    sequential.add(tf.layers.dropout({
      rate: 0.2
    }));
    
    sequential.add(tf.layers.dense({
      units: 16,
      activation: 'relu'
    }));
    
    // Output layer - binary classification
    sequential.add(tf.layers.dense({
      units: 1,
      activation: 'sigmoid'
    }));
    
    // Compile model
    sequential.compile({
      optimizer: tf.train.adam(0.0005),
      loss: 'binaryCrossentropy',
      metrics: ['accuracy']
    });
    
    // Assign to model variable
    model = sequential;
    
    console.log('ModelPredictionWorker: Created model');
  } catch (error) {
    console.error('ModelPredictionWorker: Error creating model', error);
    throw error;
  }
}

/**
 * Process prediction request
 */
function handlePredict(message: any): void {
  try {
    if (!isInitialized || !model) {
      throw new Error('Model not initialized');
    }
    
    const { features, id } = message;
    
    // Make prediction
    const prediction = predictWithModel(features);
    
    // Send result back to main thread
    const response: PredictionResultMessage = {
      type: 'prediction',
      isIntentional: prediction.isIntentional,
      confidence: prediction.confidence,
      keypointName: features.keypoint,
      id
    };
    
    self.postMessage(response);
  } catch (error) {
    console.error('ModelPredictionWorker: Error making prediction', error);
    
    // Send error response
    self.postMessage({
      type: 'error',
      error: String(error),
      id: message.id
    });
  }
}

/**
 * Run model inference
 */
function predictWithModel(features: MovementFeatures): {isIntentional: boolean, confidence: number} {
  // Use tf.tidy for automatic memory cleanup
  return tf.tidy(() => {
    // Transform features to match model input format
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
    
    // Create input tensor
    const inputTensor = tf.tensor2d([featureArray]);
    
    // Perform prediction
    const predictionOutput = model!.predict(inputTensor);
    
    // Handle different output types
    let outputTensor;
    if (Array.isArray(predictionOutput)) {
      outputTensor = predictionOutput[0];
    } else {
      outputTensor = predictionOutput;
    }
    
    // Get prediction value
    const predictionValue = outputTensor.dataSync()[0];
    
    // Return prediction result
    return {
      isIntentional: predictionValue > 0.65, // Threshold
      confidence: predictionValue
    };
  });
}

/**
 * Clean up resources
 */
function handleCleanup(message: any): void {
  try {
    // Dispose model and tensors
    if (model) {
      model.dispose();
      model = null;
    }
    
    // Force garbage collection
    tf.disposeVariables();
    
    console.log('ModelPredictionWorker: Cleaned up');
    
    // Send success response
    self.postMessage({
      type: 'cleaned',
      id: message.id
    });
  } catch (error) {
    console.error('ModelPredictionWorker: Error cleaning up', error);
    
    // Send error response
    self.postMessage({
      type: 'error',
      error: String(error),
      id: message.id
    });
  }
}

// Inform main thread that worker is ready
self.postMessage({ type: 'ready' });