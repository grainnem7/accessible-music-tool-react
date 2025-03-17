// src/services/azureCustomVisionService.ts

import { azureConfig } from '../config';

// Import directly from your project - this assumes the interface exists
// If it doesn't, uncomment the interface definition below
/* 
interface MovementFeatures {
  keypoint: string;
  velocityX: number;
  velocityY: number;
  acceleration: number;
  jitter: number;
  isSmooth: boolean;
  direction: string;
  timestamp: number;
  magnitudeOfMovement: number;
  durationOfMovement: number;
  isReversing: boolean;
  frequencyOfMovement: number;
  steadiness: number;
  patternScore: number;
  continuity: number;
}
*/

export interface IntentionalityPrediction {
  isIntentional: boolean;
  confidence: number;
}

export class AzureCustomVisionService {
  private predictionKey: string;
  private predictionEndpoint: string;
  private projectId: string;
  private publishedModelName: string;

  constructor() {
    this.predictionKey = azureConfig.customVisionKey;
    this.predictionEndpoint = azureConfig.customVisionEndpoint;
    this.projectId = azureConfig.customVisionProjectId;
    this.publishedModelName = azureConfig.customVisionModelName;
  }

  async predictIntentionality(features: any): Promise<IntentionalityPrediction> {
    try {
      const payload = {
        // Flatten movement features for Azure Custom Vision
        velocityX: features.velocityX,
        velocityY: features.velocityY,
        acceleration: features.acceleration,
        jitter: features.jitter,
        isSmooth: features.isSmooth ? 1 : 0,
        direction: features.direction,
        magnitudeOfMovement: features.magnitudeOfMovement,
        durationOfMovement: features.durationOfMovement,
        isReversing: features.isReversing ? 1 : 0,
        frequencyOfMovement: features.frequencyOfMovement,
        steadiness: features.steadiness,
        patternScore: features.patternScore,
        continuity: features.continuity
      };

      const response = await fetch(
        `${this.predictionEndpoint}/customvision/v3.0/Prediction/${this.projectId}/classify/iterations/${this.publishedModelName}`,
        {
          method: 'POST',
          headers: {
            'Prediction-Key': this.predictionKey,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(payload)
        }
      );

      if (!response.ok) {
        throw new Error(`Azure Custom Vision API error: ${response.status}`);
      }

      const result = await response.json();
      
      // Find the highest probability prediction
      const predictions = result.predictions;
      let highestProbPrediction = predictions[0];
      
      for (const prediction of predictions) {
        if (prediction.probability > highestProbPrediction.probability) {
          highestProbPrediction = prediction;
        }
      }
      
      return {
        isIntentional: highestProbPrediction.tagName === 'intentional',
        confidence: highestProbPrediction.probability
      };
    } catch (error) {
      console.error('Error calling Azure Custom Vision API:', error);
      // Return a default prediction if API call fails
      return { isIntentional: false, confidence: 0.5 };
    }
  }

  // Method to upload training data to Azure Custom Vision
  async uploadTrainingData(
    calibrationSamples: { features: any, isIntentional: boolean }[]
  ): Promise<boolean> {
    try {
      const createImageRequests = calibrationSamples.map(sample => {
        return {
          name: `movement_${Date.now()}_${Math.random().toString(36).substring(7)}`,
          features: {
            velocityX: sample.features.velocityX,
            velocityY: sample.features.velocityY,
            acceleration: sample.features.acceleration,
            jitter: sample.features.jitter,
            isSmooth: sample.features.isSmooth ? 1 : 0,
            direction: sample.features.direction,
            magnitudeOfMovement: sample.features.magnitudeOfMovement,
            durationOfMovement: sample.features.durationOfMovement,
            isReversing: sample.features.isReversing ? 1 : 0,
            frequencyOfMovement: sample.features.frequencyOfMovement,
            steadiness: sample.features.steadiness,
            patternScore: sample.features.patternScore,
            continuity: sample.features.continuity
          },
          tagIds: [sample.isIntentional ? 
            azureConfig.intentionalTagId : 
            azureConfig.unintentionalTagId]
        };
      });

      const response = await fetch(
        `${this.predictionEndpoint}/customvision/v3.0/Training/${this.projectId}/images`,
        {
          method: 'POST',
          headers: {
            'Training-Key': azureConfig.customVisionTrainingKey,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(createImageRequests)
        }
      );

      if (!response.ok) {
        throw new Error(`Azure Custom Vision upload error: ${response.status}`);
      }

      return true;
    } catch (error) {
      console.error('Error uploading to Azure Custom Vision:', error);
      return false;
    }
  }

  // Train the model with the uploaded data
  async trainModel(): Promise<boolean> {
    try {
      const response = await fetch(
        `${this.predictionEndpoint}/customvision/v3.0/Training/${this.projectId}/train`,
        {
          method: 'POST',
          headers: {
            'Training-Key': azureConfig.customVisionTrainingKey,
            'Content-Type': 'application/json'
          }
        }
      );

      if (!response.ok) {
        throw new Error(`Azure Custom Vision training error: ${response.status}`);
      }

      return true;
    } catch (error) {
      console.error('Error training Azure Custom Vision model:', error);
      return false;
    }
  }
}