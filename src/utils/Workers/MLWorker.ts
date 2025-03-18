/**
 * ML Worker Entry Point
 * This file coordinates the specialized workers for ML tasks
 */

import { InitMessage, WorkerMessage, CleanupMessage } from "../IntentionDetection/DetectionTypes";

// Import worker types for type checking

  
  // Message handling configuration
  // Rename 'WorkerType' to avoid conflict with DOM's WorkerType
  type MLWorkerType = 'featureWorker' | 'predictionWorker' | 'trainingWorker';
  
  let activeWorkers: Record<MLWorkerType, Worker | null> = {
    featureWorker: null,
    predictionWorker: null,
    trainingWorker: null
  };
  
  // Track message IDs for correlation
  let pendingMessages: Record<string, {
    resolve: (value: any) => void,
    reject: (reason: any) => void,
    timeout: number
  }> = {};
  
  // Constants
  const MESSAGE_TIMEOUT_MS = 10000; // 10 seconds
  
  // Initialize workers on startup
  self.addEventListener('message', (event: MessageEvent) => {
    const message = event.data;
    
    switch (message.type) {
      case 'init':
        handleInit(message);
        break;
        
      case 'processFeatures':
        forwardToWorker('featureWorker', message);
        break;
        
      case 'predict':
        forwardToWorker('predictionWorker', message);
        break;
        
      case 'train':
        forwardToWorker('trainingWorker', message);
        break;
        
      case 'saveModel':
        handleSaveModel(message);
        break;
        
      case 'loadModel':
        handleLoadModel(message);
        break;
        
      case 'cleanup':
        handleCleanup(message);
        break;
        
      default:
        console.error('MLWorker: Unknown message type', message.type);
    }
  });
  
  /**
   * Initialize all sub-workers
   */
  function handleInit(message: InitMessage): void {
    // Track the initialization promise
    const messageId = message.id || generateMessageId();
    let initPromises: Promise<any>[] = [];
    
    try {
      // Initialize feature extraction worker
      initPromises.push(initializeWorker('featureWorker', './FeatureExtractionWorker.js', message));
      
      // Initialize prediction worker with TensorFlow
      initPromises.push(initializeWorker('predictionWorker', './ModelPredictionWorker.js', message));
      
      // If training is requested, initialize the training worker
      if (message.includeTraining) {
        initPromises.push(initializeWorker('trainingWorker', './ModelTrainingWorker.js', message));
      }
      
      // Wait for all workers to initialize
      Promise.all(initPromises)
        .then(() => {
          self.postMessage({
            type: 'initialized',
            id: messageId,
            workersReady: Object.entries(activeWorkers)
              .filter(([_, worker]) => worker !== null)
              .map(([type]) => type)
          });
        })
        .catch((error) => {
          console.error('MLWorker: Error initializing workers', error);
          self.postMessage({
            type: 'error',
            error: String(error),
            id: messageId
          });
        });
    } catch (error) {
      console.error('MLWorker: Error during initialization', error);
      self.postMessage({
        type: 'error',
        error: String(error),
        id: messageId
      });
    }
  }
  
  /**
   * Initialize a specific worker
   */
  function initializeWorker(
    type: MLWorkerType, 
    path: string, 
    message: InitMessage
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        // Create worker
        const worker = new Worker(path);
        activeWorkers[type] = worker;
        
        // Handle worker messages
        worker.addEventListener('message', (event: MessageEvent) => {
          // Initialization message
          if (event.data.type === 'ready' || event.data.type === 'initialized') {
            resolve();
            return;
          }
          
          // Forward other messages to main thread
          self.postMessage(event.data);
          
          // Complete pending promise if it exists
          if (event.data.id && pendingMessages[event.data.id]) {
            const pending = pendingMessages[event.data.id];
            clearTimeout(pending.timeout);
            
            if (event.data.type === 'error') {
              pending.reject(event.data.error);
            } else {
              pending.resolve(event.data);
            }
            
            delete pendingMessages[event.data.id];
          }
        });
        
        // Handle worker errors
        worker.addEventListener('error', (error) => {
          console.error(`MLWorker: ${type} worker error`, error);
          reject(error);
        });
        
        // Send initialization message
        worker.postMessage({
            ...message,
            type: 'init'
          });
          
      } catch (error) {
        console.error(`MLWorker: Error initializing ${type} worker`, error);
        reject(error);
      }
    });
  }
  
  /**
   * Forward message to appropriate worker
   */
  function forwardToWorker(type: MLWorkerType, message: WorkerMessage): void {
    const worker = activeWorkers[type];
    
    if (!worker) {
      self.postMessage({
        type: 'error',
        error: `${type} worker not initialized`,
        id: message.id
      });
      return;
    }
    
    // Generate ID if not provided
    const messageId = message.id || generateMessageId();
    const messageWithId = { ...message, id: messageId };
    
    // Create promise for response tracking
    const responsePromise = new Promise((resolve, reject) => {
      // Set timeout to prevent hanging
      const timeout = setTimeout(() => {
        if (pendingMessages[messageId]) {
          reject(`Request to ${type} worker timed out after ${MESSAGE_TIMEOUT_MS}ms`);
          delete pendingMessages[messageId];
        }
      }, MESSAGE_TIMEOUT_MS);
      
      // Store promise callbacks
      pendingMessages[messageId] = { 
        resolve, 
        reject, 
        // Type assertion to handle the NodeJS.Timeout vs number issue
        timeout: timeout as unknown as number 
      };
    });
    
    // Handle promise completion
    responsePromise.catch(error => {
      console.error(`MLWorker: ${type} worker request failed`, error);
      self.postMessage({
        type: 'error',
        error: String(error),
        id: messageId
      });
    });
    
    // Send message to worker
    worker.postMessage(messageWithId);
  }
  
  /**
   * Handle model saving request
   */
  function handleSaveModel(message: any): void {
    const messageId = message.id || generateMessageId();
    const userId = message.userId;
    
    if (!userId) {
      self.postMessage({
        type: 'error',
        error: 'Missing userId for model saving',
        id: messageId
      });
      return;
    }
    
    // Forward to prediction worker which has model access
    if (activeWorkers.predictionWorker) {
      forwardToWorker('predictionWorker', {
        ...message,
        id: messageId
      });
    } else {
      self.postMessage({
        type: 'error',
        error: 'Prediction worker not available for model saving',
        id: messageId
      });
    }
  }
  
  /**
   * Handle model loading request
   */
  function handleLoadModel(message: any): void {
    const messageId = message.id || generateMessageId();
    const userId = message.userId;
    
    if (!userId) {
      self.postMessage({
        type: 'error',
        error: 'Missing userId for model loading',
        id: messageId
      });
      return;
    }
    
    // Forward to prediction worker which has model access
    if (activeWorkers.predictionWorker) {
      forwardToWorker('predictionWorker', {
        ...message,
        id: messageId
      });
    } else {
      self.postMessage({
        type: 'error',
        error: 'Prediction worker not available for model loading',
        id: messageId
      });
    }
  }
  
  /**
   * Handle cleanup request
   */
  function handleCleanup(message: CleanupMessage): void {
    const messageId = message.id || generateMessageId();
    
    // Clean up all active workers
    Object.entries(activeWorkers).forEach(([type, worker]) => {
      if (worker) {
        try {
          worker.postMessage({ type: 'cleanup', id: messageId });
          
          // Add event listener for response
          const handler = (event: MessageEvent) => {
            if (event.data.id === messageId) {
              worker.removeEventListener('message', handler);
            }
          };
          worker.addEventListener('message', handler);
          
          // Terminate after a delay to allow cleanup
          setTimeout(() => {
            worker.terminate();
            activeWorkers[type as MLWorkerType] = null;
          }, 500);
        } catch (error) {
          console.error(`MLWorker: Error cleaning up ${type} worker`, error);
        }
      }
    });
    
    // Clear pending messages
    Object.values(pendingMessages).forEach(pending => {
      clearTimeout(pending.timeout);
    });
    pendingMessages = {};
    
    // Inform main thread
    self.postMessage({
      type: 'cleaned',
      id: messageId
    });
  }
  
  /**
   * Generate a unique message ID
   */
  function generateMessageId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
  
  // Inform main thread that coordinator is ready
  self.postMessage({ type: 'ready' });