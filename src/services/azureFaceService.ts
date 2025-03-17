// src/services/azureFaceService.ts

import { azureConfig } from '../config';

export interface FacialLandmark {
  name: string;
  position: { x: number; y: number };
}

export interface FaceAttributes {
  age: number;
  gender: string;
  headPose: {
    pitch: number;
    roll: number;
    yaw: number;
  };
  smile: number;
  emotion?: {
    anger: number;
    contempt: number;
    disgust: number;
    fear: number;
    happiness: number;
    neutral: number;
    sadness: number;
    surprise: number;
  };
}

export interface DetectedFace {
  faceId: string;
  faceRectangle: {
    top: number;
    left: number;
    width: number;
    height: number;
  };
  faceLandmarks: Record<string, { x: number; y: number }>;
  faceAttributes?: FaceAttributes;
  timestamp: number;
}

export class AzureFaceService {
  private endpoint: string;
  private apiKey: string;

  constructor() {
    // You'll need to add these to your azureConfig
    this.endpoint = azureConfig.faceApiEndpoint;
    this.apiKey = azureConfig.faceApiKey;
  }

  /**
   * Detects faces in an image and returns facial landmarks and attributes
   */
  async detectFace(imageData: Blob): Promise<DetectedFace[] | null> {
    try {
      const formData = new FormData();
      formData.append('image', imageData);

      // Parameters for the Face API
      const params = new URLSearchParams({
        returnFaceId: 'true',
        returnFaceLandmarks: 'true',
        returnFaceAttributes: 'age,gender,headPose,smile,emotion',
        detectionModel: 'detection_01'
      });

      const response = await fetch(
        `${this.endpoint}/face/v1.0/detect?${params}`,
        {
          method: 'POST',
          headers: {
            'Ocp-Apim-Subscription-Key': this.apiKey,
          },
          body: formData
        }
      );

      if (!response.ok) {
        throw new Error(`Azure Face API error: ${response.status}`);
      }

      const data = await response.json();
      
      // Add timestamp to each face detected
      return data.map((face: any) => ({
        ...face,
        timestamp: Date.now()
      }));
    } catch (error) {
      console.error('Error calling Azure Face API:', error);
      return null;
    }
  }

  /**
   * Extracts facial expressions from the face detection result
   * Returns the dominant emotion
   */
  getDominantEmotion(face: DetectedFace): string | null {
    if (!face.faceAttributes?.emotion) return null;
    
    const emotions = face.faceAttributes.emotion;
    let dominantEmotion = 'neutral';
    let highestScore = emotions.neutral;
    
    for (const [emotion, score] of Object.entries(emotions)) {
      if (score > highestScore) {
        highestScore = score;
        dominantEmotion = emotion;
      }
    }
    
    return dominantEmotion;
  }

  /**
   * Returns head pose information which can be used for gesture recognition
   */
  getHeadPose(face: DetectedFace): { pitch: number; roll: number; yaw: number } | null {
    return face.faceAttributes?.headPose || null;
  }

  /**
   * Helper method to capture frame from video element
   */
  async captureVideoFrame(video: HTMLVideoElement): Promise<Blob> {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Could not get canvas context');
    
    ctx.drawImage(video, 0, 0);
    
    return new Promise<Blob>((resolve, reject) => {
      canvas.toBlob(blob => {
        if (blob) resolve(blob);
        else reject(new Error('Could not create blob from canvas'));
      }, 'image/jpeg', 0.95);
    });
  }
}