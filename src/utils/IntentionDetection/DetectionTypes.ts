import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs';

// Common keypoints to track across all detection systems
export const TRACKED_KEYPOINTS = [
  'left_wrist', 'right_wrist',
  'left_elbow', 'right_elbow',
  'left_shoulder', 'right_shoulder',
  'left_index', 'right_index',
  'left_thumb', 'right_thumb',
  'left_pinky', 'right_pinky',
  'nose', 'left_eye', 'right_eye'
];

// Feature data extracted from movements
export interface MovementFeatures {
  keypoint: string;
  velocityX: number;
  velocityY: number;
  acceleration: number;
  jitter: number;
  direction: string;
  isSmooth: boolean;
  timestamp: number;
  magnitudeOfMovement: number;
  durationOfMovement: number;
  isReversing: boolean;
  frequencyOfMovement: number;
  steadiness: number;
  patternScore: number;
  continuity: number;
}

// Results returned from intention detection
export interface MovementInfo {
  keypoint: string;
  isIntentional: boolean;
  velocity: number;
  direction: string;
  confidence: number;
}

// Samples used for calibration
export interface CalibrationSample {
  features: MovementFeatures;
  isIntentional: boolean;
}

// Azure configuration options
export interface AzureServiceConfig {
  apiKey: string;
  endpoint: string;
  enabled: boolean;
  defaultModelId?: string;
  location?: string;
}

// Status information about the detector
export interface DetectorStatus {
  isModelTrained: boolean;
  calibrationSamples: number;
  intentionalSamples: number;
  unintentionalSamples: number;
  isTfInitialized: boolean;
  calibrationQuality: number;
  azureEnabled: boolean;
  lastError?: string;
  memoryUsage?: number;
  isUsingWorkers?: boolean;
}

// Worker message types for communicating with web workers
export type WorkerMessageType = 
  | 'init'
  | 'processFeatures'
  | 'predict'
  | 'train'
  | 'cleanup'
  | 'features'      // Response type
  | 'prediction'    // Response type
  | 'error'         // Error response
  | 'initialized'   // Initialization response
  | 'ready'         // Ready response
  | 'cleaned'       // Cleanup response
  | 'saveModel'     // Save model request
  | 'loadModel'     // Load model request
  | 'modelSaved'    // Model saved response
  | 'modelLoaded';  // Model loaded response

// Base interface for all worker messages
export interface WorkerMessage {
  type: WorkerMessageType;
  id?: string; // For correlating responses with requests
}

// Message for processing pose data in worker
export interface ProcessFeaturesMessage extends WorkerMessage {
  type: 'processFeatures';
  poseHistory: Array<{poses: poseDetection.Pose[], timestamp: number}>;
  keypointName: string;
}

// Response from feature extraction worker
export interface FeaturesResultMessage extends WorkerMessage {
  type: 'features';
  features: MovementFeatures | null;
  keypointName: string;
}

// Message for prediction in worker
export interface PredictMessage extends WorkerMessage {
  type: 'predict';
  features: MovementFeatures;
}

// Response from prediction worker
export interface PredictionResultMessage extends WorkerMessage {
  type: 'prediction';
  isIntentional: boolean;
  confidence: number;
  keypointName: string;
}

// For initializing TensorFlow in worker
export interface InitMessage {
    id: string;
    preferredBackend?: string;
    includeTraining?: boolean;
    modelWeights?: ArrayBuffer;
    weightSpecs?: tf.io.WeightsManifestEntry[]; // <-- add this line
  }
  

// For cleanup
export interface CleanupMessage extends WorkerMessage {
  type: 'cleanup';
}

// For saving model
export interface SaveModelMessage extends WorkerMessage {
  type: 'saveModel';
  userId: string;
}

// For loading model
export interface LoadModelMessage extends WorkerMessage {
  type: 'loadModel';
  userId: string;
}

// Type guard function to check if poses array is valid
export function isValidPosesArray(poses: any): poses is poseDetection.Pose[] {
  return Array.isArray(poses) && 
         poses.length > 0 && 
         poses[0].keypoints && 
         Array.isArray(poses[0].keypoints);
}