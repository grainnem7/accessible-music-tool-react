// src/services/azureSpeechService.ts

import * as SpeechSDK from 'microsoft-cognitiveservices-speech-sdk';
import { azureConfig } from '../config';

export interface SpeechCommand {
  command: string;
  timestamp: number;
}

export class AzureSpeechService {
  private speechConfig: SpeechSDK.SpeechConfig;
  private recognizer: SpeechSDK.SpeechRecognizer | null = null;
  private isListening: boolean = false;
  private commandCallback: ((command: SpeechCommand) => void) | null = null;
  
  // List of valid commands
  private validCommands = [
    'start', 'stop', 'piano', 'drums', 'guitar', 
    'louder', 'softer', 'faster', 'slower'
  ];

  constructor() {
    this.speechConfig = SpeechSDK.SpeechConfig.fromSubscription(
      azureConfig.speechKey, 
      azureConfig.speechRegion
    );
    this.speechConfig.speechRecognitionLanguage = 'en-US';
  }

  public startListening(callback: (command: SpeechCommand) => void): void {
    if (this.isListening) return;
    
    this.commandCallback = callback;
    const audioConfig = SpeechSDK.AudioConfig.fromDefaultMicrophoneInput();
    this.recognizer = new SpeechSDK.SpeechRecognizer(this.speechConfig, audioConfig);
    
    // Process speech recognition results
    this.recognizer.recognized = (s: unknown, e: SpeechSDK.SpeechRecognitionEventArgs) => {
      if (e.result.reason === SpeechSDK.ResultReason.RecognizedSpeech) {
        const text = e.result.text.toLowerCase().trim();
        console.log(`Speech recognized: ${text}`);
        
        // Check if recognized speech contains any valid commands
        for (const command of this.validCommands) {
          if (text.includes(command)) {
            if (this.commandCallback) {
              this.commandCallback({
                command,
                timestamp: Date.now()
              });
            }
            break;
          }
        }
      }
    };
    
    // Start continuous recognition
    this.recognizer.startContinuousRecognitionAsync(
      () => {
        this.isListening = true;
        console.log('Speech recognition started');
      },
      (error: any) => {
        console.error('Error starting speech recognition:', error);
        this.isListening = false;
      }
    );
  }

  public stopListening(): void {
    if (!this.isListening || !this.recognizer) return;
    
    this.recognizer.stopContinuousRecognitionAsync(
      () => {
        this.isListening = false;
        console.log('Speech recognition stopped');
      },
      (error: any) => {
        console.error('Error stopping speech recognition:', error);
      }
    );
    
    this.recognizer = null;
    this.commandCallback = null;
  }
}