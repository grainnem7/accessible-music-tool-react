// src/services/azureVisionService.ts

import { azureConfig } from '../config';

export interface AzurePoseKeypoint {
  name: string;
  position: { x: number; y: number };
  confidence: number;
}

export interface AzurePoseDetectionResult {
  keypoints: AzurePoseKeypoint[];
  timestamp: number;
}

export class AzureVisionService {
  private endpoint: string;
  private apiKey: string;

  constructor() {
    this.endpoint = azureConfig.computerVisionEndpoint;
    this.apiKey = azureConfig.computerVisionKey;
  }

  async detectPose(imageData: Blob): Promise<AzurePoseDetectionResult | null> {
    try {
      const formData = new FormData();
      formData.append('image', imageData);

      const response = await fetch(
        `${this.endpoint}/vision/v3.2/analyze?features=bodyTracking`,
        {
          method: 'POST',
          headers: {
            'Ocp-Apim-Subscription-Key': this.apiKey,
          },
          body: formData
        }
      );

      if (!response.ok) {
        throw new Error(`Azure Vision API error: ${response.status}`);
      }

      const data = await response.json();
      
      // Transform Azure's format to our application format
      return this.transformBodyTrackingResults(data);
    } catch (error) {
      console.error('Error calling Azure Vision API:', error);
      return null;
    }
  }

  private transformBodyTrackingResults(data: any): AzurePoseDetectionResult {
    const keypoints: AzurePoseKeypoint[] = [];
    
    if (data.bodyTracking && data.bodyTracking.bodies && data.bodyTracking.bodies.length > 0) {
      const body = data.bodyTracking.bodies[0];
      
      // Map Azure keypoints to our format
      for (const [name, joint] of Object.entries(body.joints)) {
        keypoints.push({
          name: this.mapJointNameToOurFormat(name),
          position: {
            x: joint.position.x,
            y: joint.position.y
          },
          confidence: joint.confidence
        });
      }
    }
    
    return {
      keypoints,
      timestamp: Date.now()
    };
  }

  private mapJointNameToOurFormat(azureJointName: string): string {
    // Map Azure joint names to our application's joint names
    const jointMap: Record<string, string> = {
      'Head': 'nose',
      'Neck': 'neck',
      'ShoulderLeft': 'left_shoulder',
      'ShoulderRight': 'right_shoulder',
      'ElbowLeft': 'left_elbow',
      'ElbowRight': 'right_elbow',
      'WristLeft': 'left_wrist',
      'WristRight': 'right_wrist',
      // Add more mappings as needed
    };
    
    return jointMap[azureJointName] || azureJointName;
  }

  // Helper method to capture frame from video element
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