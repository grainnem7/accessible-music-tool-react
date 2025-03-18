
/**
 * Web Worker for feature extraction
 * Performs CPU-intensive feature calculations in a background thread
 */

import { ProcessFeaturesMessage, InitMessage, CleanupMessage, FeaturesResultMessage } from "../IntentionDetection/DetectionTypes";
import { FeatureExtractor } from "../IntentionDetection/FeatureExtractor";

// Create a feature extractor instance for the worker
const featureExtractor = new FeatureExtractor();

// Listen for messages from main thread
self.addEventListener('message', async (event: MessageEvent) => {
  const message = event.data;
  
  // Handle different message types
  switch (message.type) {
    case 'processFeatures':
      handleProcessFeatures(message as ProcessFeaturesMessage);
      break;
      
    case 'init':
      handleInit(message as InitMessage);
      break;
      
    case 'cleanup':
      handleCleanup(message as CleanupMessage);
      break;
      
    default:
      console.error('FeatureExtractionWorker: Unknown message type', message.type);
  }
});

/**
 * Process pose history to extract features
 */
function handleProcessFeatures(message: ProcessFeaturesMessage): void {
  try {
    const { poseHistory, keypointName, id } = message;
    
    // Extract features
    const features = featureExtractor.extractFeatures(poseHistory, keypointName);
    
    // Send result back to main thread
    const response: FeaturesResultMessage = {
      type: 'features',
      features,
      keypointName,
      id
    };
    
    self.postMessage(response);
  } catch (error) {
    console.error('FeatureExtractionWorker: Error processing features', error);
    
    // Send error response
    self.postMessage({
      type: 'error',
      error: String(error),
      id: message.id
    });
  }
}

/**
 * Initialize the worker
 */
function handleInit(message: InitMessage): void {
  try {
    console.log('FeatureExtractionWorker: Initialized');
    
    // Send success response
    self.postMessage({
      type: 'initialized',
      id: message.id
    });
  } catch (error) {
    console.error('FeatureExtractionWorker: Error initializing', error);
    
    // Send error response
    self.postMessage({
      type: 'error',
      error: String(error),
      id: message.id
    });
  }
}

/**
 * Clean up resources
 */
function handleCleanup(message: CleanupMessage): void {
  try {
    // Clear feature extractor state
    featureExtractor.clearState();
    
    console.log('FeatureExtractionWorker: Cleaned up');
    
    // Send success response
    self.postMessage({
      type: 'cleaned',
      id: message.id
    });
  } catch (error) {
    console.error('FeatureExtractionWorker: Error cleaning up', error);
    
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