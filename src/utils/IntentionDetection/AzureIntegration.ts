import axios, { AxiosRequestConfig } from 'axios';
import { AzureServiceConfig, MovementFeatures } from './DetectionTypes';
import { performanceMonitor } from '../Helpers/PerformanceMonitor';

/**
 * AzureIntegration
 * Handles communication with Azure Cognitive Services for enhanced ML capabilities
 */
export class AzureIntegration {
  private config: AzureServiceConfig;
  private retryCount = 3; // Number of retries for failed requests
  private requestTimeout = 5000; // 5 seconds
  private cachedPredictions = new Map<string, {
    result: { isIntentional: boolean, confidence: number },
    timestamp: number
  }>();
  private cacheValidityPeriod = 1000; // 1 second
  
  constructor(config: AzureServiceConfig) {
    this.config = config;
  }
  
  /**
   * Predict intention using Azure Computer Vision
   */
  public async predictIntention(
    keypointName: string,
    features: MovementFeatures
  ): Promise<{ isIntentional: boolean, confidence: number } | null> {
    // Return null if Azure not enabled or configured
    if (!this.config.enabled || !this.config.apiKey || !this.config.endpoint) {
      return null;
    }
    
    // Start performance tracking
    performanceMonitor.start('azurePrediction');
    
    try {
      // Generate cache key based on features
      const cacheKey = this.generateCacheKey(keypointName, features);
      
      // Check cache first
      const cachedResult = this.cachedPredictions.get(cacheKey);
      if (cachedResult && (Date.now() - cachedResult.timestamp < this.cacheValidityPeriod)) {
        performanceMonitor.end('azurePrediction');
        return cachedResult.result;
      }
      
      // Prepare request payload
      const payload = this.prepareFeaturePayload(keypointName, features);
      
      // Get model ID for this user
      const modelId = this.config.defaultModelId || 'default-model';
      
      // Build request config with retry and timeout
      const requestConfig: AxiosRequestConfig = {
        method: 'post',
        url: `${this.config.endpoint}/customvision/v3.0/prediction/${modelId}/classify/iterations/latest/image`,
        headers: {
          'Content-Type': 'application/json',
          'Prediction-Key': this.config.apiKey
        },
        data: payload,
        timeout: this.requestTimeout
      };
      
      // Make request with retry logic
      const response = await this.makeRequestWithRetry(requestConfig, this.retryCount);
      
      // Parse response
      if (response.data && response.data.predictions) {
        const intentionalPrediction = response.data.predictions.find(
          (p: any) => p.tagName === 'intentional'
        );
        
        const unintentionalPrediction = response.data.predictions.find(
          (p: any) => p.tagName === 'unintentional'
        );
        
        if (intentionalPrediction && unintentionalPrediction) {
          const result = {
            isIntentional: intentionalPrediction.probability > unintentionalPrediction.probability,
            confidence: Math.max(intentionalPrediction.probability, unintentionalPrediction.probability)
          };
          
          // Cache the result
          this.cachedPredictions.set(cacheKey, {
            result,
            timestamp: Date.now()
          });
          
          performanceMonitor.end('azurePrediction');
          return result;
        }
      }
      
      performanceMonitor.end('azurePrediction');
      return null;
    } catch (error) {
      console.error('Azure prediction error:', error);
      performanceMonitor.end('azurePrediction');
      return null;
    }
  }
  
  /**
   * Make HTTP request with retry logic
   */
  private async makeRequestWithRetry(
    config: AxiosRequestConfig, 
    maxRetries: number
  ): Promise<any> {
    let lastError: any = null;
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await axios(config);
      } catch (error) {
        lastError = error;
        
        // Don't retry on 4xx errors (except 429 - rate limit)
        if (axios.isAxiosError(error) && error.response) {
          const status = error.response.status;
          if (status >= 400 && status < 500 && status !== 429) {
            throw error;
          }
        }
        
        // Only retry if we have attempts left
        if (attempt < maxRetries) {
          // Wait with exponential backoff before retrying
          const delay = Math.pow(2, attempt) * 300; // 300, 600, 1200 ms
          await new Promise(resolve => setTimeout(resolve, delay));
        } else {
          throw lastError;
        }
      }
    }
  }
  
  /**
   * Generate cache key from features
   */
  private generateCacheKey(keypointName: string, features: MovementFeatures): string {
    // Create a cache key based on the most important features
    return `${keypointName}-${features.magnitudeOfMovement.toFixed(1)}-${features.direction}-${features.jitter.toFixed(1)}-${features.steadiness.toFixed(1)}`;
  }
  
  /**
   * Prepare feature payload for Azure
   */
  private prepareFeaturePayload(keypointName: string, features: MovementFeatures): any {
    return {
      features: {
        keypoint: keypointName,
        velocityVector: [features.velocityX, features.velocityY],
        velocityMagnitude: Math.sqrt(features.velocityX**2 + features.velocityY**2),
        acceleration: features.acceleration,
        jitter: features.jitter,
        direction: features.direction,
        isSmooth: features.isSmooth ? 1 : 0,
        magnitudeOfMovement: features.magnitudeOfMovement,
        durationOfMovement: features.durationOfMovement,
        isReversing: features.isReversing ? 1 : 0,
        frequencyOfMovement: features.frequencyOfMovement,
        steadiness: features.steadiness,
        patternScore: features.patternScore,
        continuity: features.continuity
      }
    };
  }
  
  /**
   * Train a new model on Azure
   */
  public async trainModel(
    userId: string, 
    samples: { features: MovementFeatures, isIntentional: boolean }[],
    progressCallback?: (progress: number) => void
  ): Promise<{ success: boolean, modelId?: string, error?: string }> {
    if (!this.config.enabled || !this.config.apiKey || !this.config.endpoint) {
      return { success: false, error: 'Azure integration not enabled' };
    }
    
    try {
      // Report progress
      if (progressCallback) progressCallback(0.1);
      
      // Step 1: Create or get project
      const project = await this.getOrCreateProject(userId);
      
      if (!project.success) {
        return { success: false, error: project.message };
      }
      
      // Report progress
      if (progressCallback) progressCallback(0.2);
      
      // Step 2: Ensure tags exist
      const tags = await this.createTags(project.projectId);
      
      if (!tags.success) {
        return { success: false, error: tags.message };
      }
      
      // Report progress
      if (progressCallback) progressCallback(0.3);
      
      // Step 3: Upload training data
      const uploadResult = await this.uploadTrainingData(
        project.projectId, 
        tags.intentionalTagId!, 
        tags.unintentionalTagId!, 
        samples,
        (progress) => {
          if (progressCallback) progressCallback(0.3 + progress * 0.4);
        }
      );
      
      if (!uploadResult.success) {
        return { success: false, error: uploadResult.message };
      }
      
      // Report progress
      if (progressCallback) progressCallback(0.7);
      
      // Step 4: Train the model
      const trainingResult = await this.trainAzureModel(project.projectId);
      
      if (!trainingResult.success) {
        return { 
          success: false, 
          error: trainingResult.message
        };
      }
      
      // Report progress
      if (progressCallback) progressCallback(0.9);
      
      // Step 5: Publish the trained model
      const publishResult = await this.publishModel(
        project.projectId,
        trainingResult.iterationId!,
        userId
      );
      
      if (!publishResult.success) {
        return { 
          success: false, 
          error: publishResult.message
        };
      }
      
      // Store the model ID for future use
      this.config.defaultModelId = project.projectId;
      localStorage.setItem(`azure-model-id-${userId}`, project.projectId);
      
      // Report completion
      if (progressCallback) progressCallback(1.0);
      
      return {
        success: true,
        modelId: project.projectId
      };
    } catch (error) {
      console.error('Azure training error:', error);
      return {
        success: false,
        error: String(error)
      };
    }
  }
  
  /**
   * Create or retrieve a project for the user
   */
  private async getOrCreateProject(userId: string): Promise<{
    success: boolean,
    projectId: string,
    message: string
  }> {
    try {
      const storedProjectId = localStorage.getItem(`azure-project-id-${userId}`);
      
      if (storedProjectId) {
        // Verify the project still exists
        try {
          await axios({
            method: 'get',
            url: `${this.config.endpoint}/customvision/v3.0/training/projects/${storedProjectId}`,
            headers: {
              'Training-Key': this.config.apiKey
            },
            timeout: this.requestTimeout
          });
          
          // Project exists, return it
          return {
            success: true,
            projectId: storedProjectId,
            message: 'Retrieved existing project'
          };
        } catch (error) {
          // Project not found or other error, continue to create a new one
          console.warn('Stored project not found, creating new project:', error);
        }
      }
      
      // Create a new project
      const projectName = `intention-detector-${userId}-${Date.now()}`;
      
      const response = await axios({
        method: 'post',
        url: `${this.config.endpoint}/customvision/v3.0/training/projects`,
        headers: {
          'Training-Key': this.config.apiKey,
          'Content-Type': 'application/json'
        },
        data: {
          name: projectName,
          description: `Movement intention detection model for user ${userId}`,
          domainId: 'Multiclass', // Use general domain for classification
          classificationType: 'Multiclass'
        },
        timeout: this.requestTimeout
      });
      
      if (response.data && response.data.id) {
        const projectId = response.data.id;
        
        // Save for future use
        localStorage.setItem(`azure-project-id-${userId}`, projectId);
        
        return {
          success: true,
          projectId,
          message: 'Created new project'
        };
      } else {
        throw new Error('Invalid response from Azure when creating project');
      }
    } catch (error) {
      console.error('Error creating/retrieving Azure project:', error);
      return {
        success: false,
        projectId: '',
        message: `Error: ${error}`
      };
    }
  }
  
  /**
   * Create required tags or retrieve existing ones
   */
  private async createTags(projectId: string): Promise<{
    success: boolean,
    intentionalTagId?: string,
    unintentionalTagId?: string,
    message: string
  }> {
    try {
      // Get existing tags
      const tagsResponse = await axios({
        method: 'get',
        url: `${this.config.endpoint}/customvision/v3.0/training/projects/${projectId}/tags`,
        headers: {
          'Training-Key': this.config.apiKey
        },
        timeout: this.requestTimeout
      });
      
      let intentionalTag = tagsResponse.data.find((t: any) => t.name === 'intentional');
      let unintentionalTag = tagsResponse.data.find((t: any) => t.name === 'unintentional');
      
      // Create tags if they don't exist
      if (!intentionalTag) {
        const intentionalResponse = await axios({
          method: 'post',
          url: `${this.config.endpoint}/customvision/v3.0/training/projects/${projectId}/tags`,
          headers: {
            'Training-Key': this.config.apiKey,
            'Content-Type': 'application/json'
          },
          data: { name: 'intentional' },
          timeout: this.requestTimeout
        });
        
        intentionalTag = intentionalResponse.data;
      }
      
      if (!unintentionalTag) {
        const unintentionalResponse = await axios({
          method: 'post',
          url: `${this.config.endpoint}/customvision/v3.0/training/projects/${projectId}/tags`,
          headers: {
            'Training-Key': this.config.apiKey,
            'Content-Type': 'application/json'
          },
          data: { name: 'unintentional' },
          timeout: this.requestTimeout
        });
        
        unintentionalTag = unintentionalResponse.data;
      }
      
      return {
        success: true,
        intentionalTagId: intentionalTag.id,
        unintentionalTagId: unintentionalTag.id,
        message: 'Tags created successfully'
      };
    } catch (error) {
      console.error('Error creating Azure tags:', error);
      return {
        success: false,
        message: `Error creating tags: ${error}`
      };
    }
  }
  
  /**
   * Upload training data to Azure
   */
  private async uploadTrainingData(
    projectId: string,
    intentionalTagId: string,
    unintentionalTagId: string,
    samples: { features: MovementFeatures, isIntentional: boolean }[],
    progressCallback?: (progress: number) => void
  ): Promise<{ success: boolean, message: string }> {
    try {
      // Batch samples to avoid overwhelming the API
      const BATCH_SIZE = 50;
      const batches = [];
      
      // Create batches
      for (let i = 0; i < samples.length; i += BATCH_SIZE) {
        batches.push(samples.slice(i, i + BATCH_SIZE));
      }
      
      // Process each batch
      for (let i = 0; i < batches.length; i++) {
        const batch = batches[i];
        
        const batchData = {
          images: batch.map(sample => ({
            name: `${sample.features.keypoint}-${Date.now()}-${Math.random().toString(36).substring(2, 8)}`,
            tagIds: [sample.isIntentional ? intentionalTagId : unintentionalTagId],
            regions: [],
            metadata: {
              featureData: JSON.stringify(sample.features)
            }
          }))
        };
        
        // Upload batch
        await axios({
          method: 'post',
          url: `${this.config.endpoint}/customvision/v3.0/training/projects/${projectId}/images/batch`,
          headers: {
            'Training-Key': this.config.apiKey,
            'Content-Type': 'application/json'
          },
          data: batchData,
          timeout: 30000 // Longer timeout for uploads
        });
        
        // Update progress
        if (progressCallback) {
          progressCallback((i + 1) / batches.length);
        }
      }
      
      return {
        success: true,
        message: `Uploaded ${samples.length} samples`
      };
    } catch (error) {
      console.error('Error uploading training data:', error);
      return {
        success: false,
        message: `Error uploading training data: ${error}`
      };
    }
  }
  
  /**
   * Train the model on Azure
   */
  private async trainAzureModel(projectId: string): Promise<{
    success: boolean,
    iterationId?: string,
    message: string
  }> {
    try {
      // Start training
      const trainResponse = await axios({
        method: 'post',
        url: `${this.config.endpoint}/customvision/v3.0/training/projects/${projectId}/train`,
        headers: {
          'Training-Key': this.config.apiKey,
          'Content-Type': 'application/json'
        },
        data: {
          name: `model-${Date.now()}`,
          trainingType: 'Regular'
        },
        timeout: this.requestTimeout
      });
      
      const iterationId = trainResponse.data.id;
      
      // Check training status periodically until complete
      let isTrainingComplete = false;
      let trainingCheckAttempts = 0;
      
      while (!isTrainingComplete && trainingCheckAttempts < 60) { // Wait up to 10 minutes (10s * 60)
        trainingCheckAttempts++;
        
        // Wait 10 seconds between checks
        await new Promise(resolve => setTimeout(resolve, 10000));
        
        // Check status
        const statusResponse = await axios({
          method: 'get',
          url: `${this.config.endpoint}/customvision/v3.0/training/projects/${projectId}/iterations/${iterationId}`,
          headers: {
            'Training-Key': this.config.apiKey
          },
          timeout: this.requestTimeout
        });
        
        if (statusResponse.data.status === 'Completed') {
          isTrainingComplete = true;
        } else if (statusResponse.data.status === 'Failed') {
          throw new Error('Azure training failed: ' + (statusResponse.data.statusMessage || 'Unknown error'));
        }
      }
      
      if (!isTrainingComplete) {
        throw new Error('Azure training timed out');
      }
      
      return {
        success: true,
        iterationId,
        message: 'Training completed successfully'
      };
    } catch (error) {
      console.error('Error training Azure model:', error);
      return {
        success: false,
        message: `Error training model: ${error}`
      };
    }
  }
  
  /**
   * Publish the trained model for use
   */
  private async publishModel(
    projectId: string,
    iterationId: string,
    userId: string
  ): Promise<{ success: boolean, message: string }> {
    try {
      await axios({
        method: 'patch',
        url: `${this.config.endpoint}/customvision/v3.0/training/projects/${projectId}/iterations/${iterationId}/publish`,
        headers: {
          'Training-Key': this.config.apiKey,
          'Content-Type': 'application/json'
        },
        data: {
          publishName: `model-${userId}`,
          predictionId: projectId
        },
        timeout: this.requestTimeout
      });
      
      return {
        success: true,
        message: 'Model published successfully'
      };
    } catch (error) {
      console.error('Error publishing Azure model:', error);
      return {
        success: false,
        message: `Error publishing model: ${error}`
      };
    }
  }
  
  /**
   * Update configuration
   */
  public updateConfig(config: Partial<AzureServiceConfig>): void {
    this.config = {
      ...this.config,
      ...config
    };
  }
  
  /**
   * Clean up resources
   */
  public dispose(): void {
    this.cachedPredictions.clear();
  }
}