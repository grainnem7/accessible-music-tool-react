// src/services/azureMLService.ts

import { azureConfig } from '../config';

export interface MLPrediction {
  isIntentional: boolean;
  confidence: number;
}

export class AzureMLService {
  private endpoint: string;
  private apiKey: string;

  constructor() {
    this.endpoint = azureConfig.mlEndpoint;
    this.apiKey = azureConfig.mlApiKey;
  }

  async predictIntentionality(features: any): Promise<MLPrediction> {
    try {
      // Prepare the data in the format expected by your Azure ML model
      const payload = {
        data: [
          {
            velocityX: features.velocityX,
            velocityY: features.velocityY,
            acceleration: features.acceleration,
            jitter: features.jitter,
            isSmooth: features.isSmooth ? 1 : 0,
            direction_up: features.direction === 'up' ? 1 : 0,
            direction_down: features.direction === 'down' ? 1 : 0,
            direction_left: features.direction === 'left' ? 1 : 0,
            direction_right: features.direction === 'right' ? 1 : 0,
            magnitudeOfMovement: features.magnitudeOfMovement,
            durationOfMovement: features.durationOfMovement,
            isReversing: features.isReversing ? 1 : 0,
            frequencyOfMovement: features.frequencyOfMovement,
            steadiness: features.steadiness,
            patternScore: features.patternScore,
            continuity: features.continuity
          }
        ]
      };

      const response = await fetch(
        this.endpoint,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${this.apiKey}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(payload)
        }
      );

      if (!response.ok) {
        throw new Error(`Azure ML API error: ${response.status}`);
      }

      const result = await response.json();
      
      // Parse the Azure ML response
      // The exact format will depend on your deployed model
      return {
        isIntentional: result.predictions[0] > 0.5,
        confidence: Math.abs(result.predictions[0] - 0.5) * 2
      };
    } catch (error) {
      console.error('Error calling Azure ML API:', error);
      return { isIntentional: false, confidence: 0.5 };
    }
  }

  // Method to train or retrain an Azure ML model with new data
  async trainModel(
    userId: string, 
    calibrationSamples: { features: any, isIntentional: boolean }[]
  ): Promise<boolean> {
    try {
      // Prepare the training data
      const trainingData = calibrationSamples.map(sample => ({
        userId,
        features: {
          velocityX: sample.features.velocityX,
          velocityY: sample.features.velocityY,
          acceleration: sample.features.acceleration,
          jitter: sample.features.jitter,
          isSmooth: sample.features.isSmooth ? 1 : 0,
          direction_up: sample.features.direction === 'up' ? 1 : 0,
          direction_down: sample.features.direction === 'down' ? 1 : 0,
          direction_left: sample.features.direction === 'left' ? 1 : 0,
          direction_right: sample.features.direction === 'right' ? 1 : 0,
          magnitudeOfMovement: sample.features.magnitudeOfMovement,
          durationOfMovement: sample.features.durationOfMovement,
          isReversing: sample.features.isReversing ? 1 : 0,
          frequencyOfMovement: sample.features.frequencyOfMovement,
          steadiness: sample.features.steadiness,
          patternScore: sample.features.patternScore,
          continuity: sample.features.continuity
        },
        label: sample.isIntentional ? 1 : 0
      }));

      // Call Azure ML training endpoint
      const response = await fetch(
        `${azureConfig.mlTrainingEndpoint}/train/${userId}`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${this.apiKey}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ trainingData })
        }
      );

      if (!response.ok) {
        throw new Error(`Azure ML training API error: ${response.status}`);
      }

      const result = await response.json();
      return result.success === true;
    } catch (error) {
      console.error('Error training Azure ML model:', error);
      return false;
    }
  }
}