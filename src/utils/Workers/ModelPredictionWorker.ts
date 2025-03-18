import * as tf from '@tensorflow/tfjs';
import { InitMessage, PredictMessage, CleanupMessage, PredictionResultMessage, MovementFeatures } from '../IntentionDetection/DetectionTypes';

let model: tf.Sequential | null = null;
let isInitialized = false;

const startTime = Date.now();

self.addEventListener('message', async (event: MessageEvent) => {
  const message = event.data;

  switch (message.type) {
    case 'init':
      await handleInit(message as InitMessage);
      break;

    case 'predict':
      handlePredict(message as PredictMessage);
      break;

    case 'cleanup':
      handleCleanup(message as CleanupMessage);
      break;

    default:
      console.error('Unknown message type', message.type);
  }
});

async function handleInit(message: InitMessage): Promise<void> {
  try {
    await tf.ready();

    if (message.preferredBackend) {
      await tf.setBackend(message.preferredBackend);
      console.log(`Backend set to ${tf.getBackend()}`);
    }

    createModel();

    if (message.modelWeights && message.weightSpecs) {
      const weightMap = tf.io.decodeWeights(message.modelWeights, message.weightSpecs);
      await model!.loadWeights(weightMap);
      Object.values(weightMap).forEach(tensor => tensor.dispose());
      console.log('Model weights loaded');
    }

    isInitialized = true;

    self.postMessage({
      type: 'initialized',
      id: message.id,
      backend: tf.getBackend(),
      initTimeMs: Date.now() - startTime,
    });
  } catch (error) {
    console.error('Initialization error:', error);
    self.postMessage({
      type: 'error',
      error: String(error),
      id: message.id,
    });
  }
}

function createModel(): void {
  model = tf.sequential();

  model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [15] }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

  model.compile({ optimizer: tf.train.adam(0.0005), loss: 'binaryCrossentropy', metrics: ['accuracy'] });

  console.log('Model created');
}

function handlePredict(message: PredictMessage): void {
  try {
    if (!isInitialized || !model) throw new Error('Model not initialized');

    const prediction = predictWithModel(message.features);

    const response: PredictionResultMessage = {
      type: 'prediction',
      isIntentional: prediction.isIntentional,
      confidence: prediction.confidence,
      keypointName: message.features.keypoint,
      id: message.id,
    };

    self.postMessage(response);
  } catch (error) {
    console.error('Prediction error:', error);
    self.postMessage({ type: 'error', error: String(error), id: message.id });
  }
}

function predictWithModel(features: MovementFeatures): { isIntentional: boolean; confidence: number } {
  return tf.tidy(() => {
    const featureArray = [
      features.velocityX,
      features.velocityY,
      features.acceleration,
      features.jitter,
      features.isSmooth ? 1 : 0,
      features.direction === 'up' ? 1 : 0,
      features.direction === 'down' ? 1 : 0,
      features.direction === 'left' ? 1 : 0,
      features.direction === 'right' ? 1 : 0,
      features.magnitudeOfMovement / 100,
      features.isReversing ? 1 : 0,
      features.frequencyOfMovement / 10,
      features.steadiness,
      features.patternScore,
      features.continuity,
    ];

    const inputTensor = tf.tensor2d([featureArray]);
    const outputTensor = model!.predict(inputTensor) as tf.Tensor;
    const predictionValue = outputTensor.dataSync()[0];

    return {
      isIntentional: predictionValue > 0.65,
      confidence: predictionValue,
    };
  });
}

function handleCleanup(message: CleanupMessage): void {
  try {
    if (model) {
      model.dispose();
      model = null;
    }
    tf.disposeVariables();

    self.postMessage({ type: 'cleaned', id: message.id });
    console.log('Cleaned up');
  } catch (error) {
    console.error('Cleanup error:', error);
    self.postMessage({ type: 'error', error: String(error), id: message.id });
  }
}

self.postMessage({ type: 'ready' });
