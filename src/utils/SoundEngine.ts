import * as Tone from 'tone';

export interface SoundMapping {
  keypoint: string;
  direction: string;
  soundType: 'note' | 'chord' | 'drum' | 'effect';
  soundValue: string | string[];
  parameter?: 'volume' | 'pitch' | 'tempo' | 'filter';
}

export interface SoundPreset {
  name: string;
  description: string;
  instrument: string;
  mappings: SoundMapping[];
}

export class SoundEngine {
  private synth: Tone.PolySynth;
  private drums: Tone.Sampler;
  private effectSynth: Tone.FMSynth;
  private reverb: Tone.Reverb;
  private filter: Tone.Filter;
  private mappings: SoundMapping[] = [];
  private notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  private activeNotes: Map<string, string> = new Map(); // Track currently active notes
  private isInitialized = false;
  private initializationPromise: Promise<void> | null = null;

  constructor() {
    // Initialize Tone.js instruments
    this.synth = new Tone.PolySynth(Tone.Synth).toDestination();
    
    // Create effects chain
    this.reverb = new Tone.Reverb(2).toDestination();
    this.filter = new Tone.Filter(1000, "lowpass").connect(this.reverb);
    this.synth.connect(this.filter);
    
    // Initialize drum sampler
    this.drums = new Tone.Sampler({
      "C2": "https://tonejs.github.io/audio/drum-samples/808/kick.mp3",
      "D2": "https://tonejs.github.io/audio/drum-samples/808/snare.mp3",
      "E2": "https://tonejs.github.io/audio/drum-samples/808/hihat.mp3",
      "F2": "https://tonejs.github.io/audio/drum-samples/808/clap.mp3",
    }).connect(this.reverb);
    
    // FM synth for effects
    this.effectSynth = new Tone.FMSynth().connect(this.filter);
    
    // Set default synth parameters
    this.synth.set({
      envelope: {
        attack: 0.02,
        decay: 0.1,
        sustain: 0.3,
        release: 1
      }
    });
  }

  // Initialize the audio context (must be called from a user action like a button click)
  public async initialize(): Promise<void> {
    // If already initializing, return the existing promise
    if (this.initializationPromise) {
      return this.initializationPromise;
    }
    
    this.initializationPromise = new Promise<void>(async (resolve, reject) => {
      try {
        if (!this.isInitialized || Tone.context.state !== 'running') {
          console.log('Starting Tone.js audio context...');
          
          // First try to resume if the context already exists but is suspended
          if (Tone.context.state === 'suspended') {
            await Tone.context.resume();
            console.log('Resumed existing audio context');
          }
          
          // Start Tone.js audio context if needed
          await Tone.start();
          console.log('Tone.js started');
          
          // Double check if it's really running
          if (Tone.context.state !== 'running') {
            console.log('Audio context still not running, trying again...');
            await Tone.context.resume();
          }
          
          console.log('Tone.js audio context state:', Tone.context.state);
          
          // Play a silent note to fully initialize audio
          setTimeout(() => {
            try {
              this.synth.triggerAttackRelease("C4", 0.01, Tone.now(), 0.01);
              console.log('Initialization note played');
            } catch (e) {
              console.error('Error playing initialization note:', e);
            }
          }, 100);
          
          // Set up instruments
          this.synth.toDestination();
          this.drums.toDestination();
          this.effectSynth.toDestination();
          
          this.isInitialized = true;
          console.log('Audio fully initialized');
        }
        
        resolve();
      } catch (error) {
        console.error('Error initializing audio:', error);
        this.initializationPromise = null;
        reject(error);
      } finally {
        // Reset promise if completed
        setTimeout(() => {
          this.initializationPromise = null;
        }, 1000);
      }
    });
    
    return this.initializationPromise;
  }
  
  // Set the gesture-to-sound mappings
  public setMappings(mappings: SoundMapping[]): void {
    this.mappings = mappings;
  }
  
  // Get available presets
  public getPresets(): SoundPreset[] {
    return [
      {
        name: 'Piano',
        description: 'Classic piano sounds with right hand for melody, left for chords',
        instrument: 'piano',
        mappings: [
          { keypoint: 'right_wrist', direction: 'up', soundType: 'note', soundValue: 'C4' },
          { keypoint: 'left_wrist', direction: 'up', soundType: 'chord', soundValue: ['C3', 'E3', 'G3'] },
          { keypoint: 'right_wrist', direction: 'down', soundType: 'note', soundValue: 'G4' },
          { keypoint: 'left_wrist', direction: 'down', soundType: 'chord', soundValue: ['F3', 'A3', 'C4'] },
          { keypoint: 'right_wrist', direction: 'left', soundType: 'note', soundValue: 'E4' },
          { keypoint: 'right_wrist', direction: 'right', soundType: 'note', soundValue: 'D4' }
        ]
      },
      {
        name: 'Drums',
        description: 'Percussion kit with different hand movements',
        instrument: 'drums',
        mappings: [
          { keypoint: 'right_wrist', direction: 'down', soundType: 'drum', soundValue: 'C2' }, // Kick
          { keypoint: 'left_wrist', direction: 'down', soundType: 'drum', soundValue: 'D2' }, // Snare
          { keypoint: 'right_wrist', direction: 'right', soundType: 'drum', soundValue: 'E2' }, // Hi-hat
          { keypoint: 'left_wrist', direction: 'right', soundType: 'drum', soundValue: 'F2' }, // Clap
          { keypoint: 'right_wrist', direction: 'up', soundType: 'drum', soundValue: 'E2' }, // Hi-hat
          { keypoint: 'left_wrist', direction: 'up', soundType: 'drum', soundValue: 'F2' }  // Clap
        ]
      },
      {
        name: 'Theremin',
        description: 'Continuous sound control like a theremin',
        instrument: 'theremin',
        mappings: [
          { 
            keypoint: 'right_wrist', 
            direction: 'up', 
            soundType: 'note', 
            soundValue: 'C4', 
            parameter: 'pitch'
          },
          { 
            keypoint: 'left_wrist', 
            direction: 'right', 
            soundType: 'note', 
            soundValue: 'C4', 
            parameter: 'volume'
          },
          { 
            keypoint: 'right_wrist', 
            direction: 'down', 
            soundType: 'note', 
            soundValue: 'G3'
          },
          { 
            keypoint: 'left_wrist', 
            direction: 'left', 
            soundType: 'note', 
            soundValue: 'E3'
          }
        ]
      },
      {
        name: 'Electronic',
        description: 'Synthetic sounds and effects for electronic music',
        instrument: 'electronic',
        mappings: [
          { keypoint: 'right_wrist', direction: 'up', soundType: 'note', soundValue: 'E3' },
          { keypoint: 'left_wrist', direction: 'up', soundType: 'effect', soundValue: 'sweep' },
          { keypoint: 'right_wrist', direction: 'down', soundType: 'effect', soundValue: 'wobble' },
          { keypoint: 'nose', direction: 'right', soundType: 'note', soundValue: 'G2', parameter: 'filter' },
          { keypoint: 'right_wrist', direction: 'left', soundType: 'effect', soundValue: 'arpeggio' },
          { keypoint: 'left_wrist', direction: 'left', soundType: 'effect', soundValue: 'shimmer' }
        ]
      },
      {
        name: 'Orchestral',
        description: 'Lush orchestral sounds with strings and brass',
        instrument: 'orchestral',
        mappings: [
          { keypoint: 'right_wrist', direction: 'up', soundType: 'chord', soundValue: ['C4', 'E4', 'G4', 'B4'] },
          { keypoint: 'left_wrist', direction: 'up', soundType: 'chord', soundValue: ['G3', 'B3', 'D4', 'F4'] },
          { keypoint: 'right_shoulder', direction: 'up', soundType: 'note', soundValue: 'C5' },
          { keypoint: 'left_shoulder', direction: 'up', soundType: 'note', soundValue: 'G4' },
          { keypoint: 'right_wrist', direction: 'down', soundType: 'chord', soundValue: ['D4', 'F4', 'A4'] },
          { keypoint: 'left_wrist', direction: 'down', soundType: 'chord', soundValue: ['A3', 'C4', 'E4'] }
        ]
      }
    ];
  }
  
  // Load a preset by name
  public loadPreset(presetName: string): void {
    const preset = this.getPresets().find(p => p.name === presetName);
    if (preset) {
      this.setMappings(preset.mappings);
      
      // Change synth settings based on preset
      if (preset.instrument === 'piano') {
        this.synth.set({
          envelope: {
            attack: 0.01,
            decay: 0.1,
            sustain: 0.8,
            release: 2
          }
        });
      } else if (preset.instrument === 'electronic') {
        this.synth.set({
          envelope: {
            attack: 0.05,
            decay: 0.2,
            sustain: 0.5,
            release: 0.5
          }
        });
        this.filter.frequency.value = 2000;
      } else if (preset.instrument === 'orchestral') {
        this.synth.set({
          envelope: {
            attack: 0.1,
            decay: 0.3,
            sustain: 0.7,
            release: 3
          }
        });
        this.reverb.decay = 4;
      }
      
      console.log(`Loaded preset: ${presetName}`);
    }
  }
  
  // Process a detected movement and trigger appropriate sound
  public processMovement(keypoint: string, isIntentional: boolean, direction: string, velocity: number): void {
    if (!this.isInitialized) {
      console.log("Warning: Attempting to play sound before initialization");
      // Try to initialize
      this.initialize().catch(err => console.error("Failed to initialize during processMovement", err));
      return;
    }
    
    try {
      // Find matching mappings
      const matchingMappings = this.mappings.filter(mapping => 
        mapping.keypoint === keypoint && 
        (mapping.direction === direction || mapping.direction === 'any')
      );
      
      if (matchingMappings.length === 0) {
        // If no exact match, try to find any mapping for this keypoint
        const anyDirectionMappings = this.mappings.filter(mapping => 
          mapping.keypoint === keypoint
        );
        
        if (anyDirectionMappings.length > 0) {
          // Take the first mapping for this keypoint
          const mapping = anyDirectionMappings[0];
          console.log(`Playing sound for ${keypoint} (using fallback mapping) with velocity ${velocity}`);
          this.playSound(mapping, velocity);
          return;
        }
        
        // If still no match, play a default sound based on keypoint type
        console.log(`Playing default sound for ${keypoint} ${direction}`);
        
        if (keypoint.includes('wrist')) {
          // Default piano notes for wrists
          const note = keypoint.includes('right') ? 'C4' : 'G3';
          this.synth.triggerAttackRelease(note, "8n", Tone.now(), Math.min(velocity / 30, 1));
        } else if (keypoint.includes('shoulder')) {
          // Default bass notes for shoulders
          const note = keypoint.includes('right') ? 'C2' : 'G2';
          this.synth.triggerAttackRelease(note, "4n", Tone.now(), Math.min(velocity / 30, 1));
        } else {
          // Default effect for other keypoints
          this.effectSynth.triggerAttackRelease("C3", "8n", Tone.now(), Math.min(velocity / 30, 1));
        }
      } else {
        // Play sounds for all matching mappings
        matchingMappings.forEach(mapping => {
          console.log(`Playing sound for ${keypoint} ${direction} with velocity ${velocity} (mapped sound)`);
          this.playSound(mapping, velocity);
        });
      }
    } catch (error) {
      console.error('Error processing movement:', error);
      // Fallback sound if error occurs
      try {
        this.synth.triggerAttackRelease("C4", "8n", Tone.now(), 0.5);
      } catch (e) {
        console.error('Even fallback sound failed:', e);
      }
    }
  }
  
  // Abstracted sound playing logic
  private playSound(mapping: SoundMapping, velocity: number): void {
    // Normalize velocity to 0-1 range for sound parameters
    const normalizedVelocity = Math.min(Math.max(velocity / 50, 0), 1);
    
    switch (mapping.soundType) {
      case 'note':
        this.playNote(mapping.soundValue as string, normalizedVelocity, mapping.parameter, mapping.keypoint);
        break;
        
      case 'chord':
        this.playChord(mapping.soundValue as string[], normalizedVelocity);
        break;
        
      case 'drum':
        this.playDrum(mapping.soundValue as string, normalizedVelocity);
        break;
        
      case 'effect':
        this.playEffect(mapping.soundValue as string, normalizedVelocity);
        break;
    }
  }
  
  // Play a single note
  private playNote(note: string, velocity: number, parameter?: string, keypoint?: string): void {
    try {
      // If this is a parameter control (like theremin mode)
      if (parameter) {
        switch (parameter) {
          case 'pitch':
            // Map y position to pitch in a scale
            const octave = Math.floor(velocity * 3) + 3; // Map to octaves 3-5
            const noteIndex = Math.floor(velocity * 12) % 12;
            const pitchNote = `${this.notes[noteIndex]}${octave}`;
            
            // If we already have a note playing for this keypoint, stop it
            const currentNote = this.activeNotes.get(keypoint || '');
            if (currentNote) {
              this.synth.triggerRelease(currentNote);
            }
            
            // Play new note
            this.synth.triggerAttack(pitchNote, Tone.now(), 0.5);
            this.activeNotes.set(keypoint || '', pitchNote);
            break;
            
          case 'volume':
            // Map position to volume
            this.synth.volume.value = Tone.gainToDb(velocity);
            break;
            
          case 'filter':
            // Map position to filter frequency
            const filterFreq = 100 + (velocity * 10000);
            this.filter.frequency.value = filterFreq;
            break;
            
          case 'tempo':
            // Map position to tempo (BPM)
            const bpm = 60 + (velocity * 120); // 60-180 BPM
            Tone.Transport.bpm.value = bpm;
            break;
        }
      } else {
        // Just play the note once
        console.log(`Playing note: ${note} with velocity ${velocity}`);
        this.synth.triggerAttackRelease(note, "8n", Tone.now(), velocity);
      }
    } catch (error) {
      console.error('Error playing note:', error, note);
    }
  }
  
  // Play a chord
  private playChord(notes: string[], velocity: number): void {
    try {
      console.log(`Playing chord: ${notes.join(', ')} with velocity ${velocity}`);
      this.synth.triggerAttackRelease(notes, "4n", Tone.now(), velocity);
    } catch (error) {
      console.error('Error playing chord:', error, notes);
    }
  }
  
  // Play a drum sound
  private playDrum(note: string, velocity: number): void {
    try {
      console.log(`Playing drum: ${note} with velocity ${velocity}`);
      this.drums.triggerAttackRelease(note, "8n", Tone.now(), velocity);
    } catch (error) {
      console.error('Error playing drum:', error, note);
    }
  }
  
  // Play a special effect
  private playEffect(effect: string, intensity: number): void {
    try {
      console.log(`Playing effect: ${effect} with intensity ${intensity}`);
      switch (effect) {
        case 'sweep':
          this.effectSynth.triggerAttackRelease("C3", "8n", Tone.now(), intensity);
          this.filter.frequency.rampTo(100 + (intensity * 5000), 0.5);
          break;
          
        case 'wobble':
          this.effectSynth.modulationIndex.value = 10 * intensity;
          this.effectSynth.triggerAttackRelease("C2", "4n", Tone.now(), intensity);
          break;
          
        case 'arpeggio':
          const notes = ['C4', 'E4', 'G4', 'B4', 'C5'];
          for (let i = 0; i < notes.length; i++) {
            this.synth.triggerAttackRelease(
              notes[i], 
              "16n", 
              Tone.now() + (i * 0.1), 
              intensity
            );
          }
          break;
          
        case 'shimmer':
          this.effectSynth.harmonicity.value = 3;
          this.reverb.decay = 4;
          this.effectSynth.triggerAttackRelease("G4", "2n", Tone.now(), intensity * 0.7);
          break;
          
        default:
          this.effectSynth.triggerAttackRelease("C4", "8n", Tone.now(), intensity);
      }
    } catch (error) {
      console.error('Error playing effect:', error, effect);
    }
  }
  
  // Stop all sounds
  public stopAllSounds(): void {
    this.synth.releaseAll();
    this.effectSynth.triggerRelease();
    this.activeNotes.clear();
  }
  
  // Set reverb amount
  public setReverbAmount(amount: number): void {
    // amount should be between 0-1
    this.reverb.decay = 1 + (amount * 4); // 1-5 seconds decay
    this.reverb.wet.value = amount;
  }
  
  // Set filter frequency
  public setFilterFrequency(frequency: number): void {
    this.filter.frequency.value = frequency;
  }
  
  // Set overall volume
  public setVolume(volume: number): void {
    // volume should be between 0-1
    this.synth.volume.value = Tone.gainToDb(volume);
    this.effectSynth.volume.value = Tone.gainToDb(volume);
    this.drums.volume.value = Tone.gainToDb(volume);
  }
  
  // Check if the audio context is running
  public isAudioRunning(): boolean {
    return this.isInitialized && Tone.context.state === 'running';
  }
  
  // Play a test tone to verify audio is working
  public playTestTone(): void {
    if (this.isInitialized) {
      console.log("Playing test tone");
      this.synth.triggerAttackRelease("C4", "8n", Tone.now(), 0.7);
    } else {
      console.log("Cannot play test tone - audio not initialized");
      this.initialize()
        .then(() => {
          this.synth.triggerAttackRelease("C4", "8n", Tone.now(), 0.7);
          console.log("Test tone played after initialization");
        })
        .catch(err => console.error("Failed to initialize for test tone", err));
    }
  }
  
  // Custom mapping creation helper
  public createCustomMapping(
    keypoint: string, 
    direction: string, 
    soundType: 'note' | 'chord' | 'drum' | 'effect',
    soundValue: string | string[],
    parameter?: 'volume' | 'pitch' | 'tempo' | 'filter'
  ): SoundMapping {
    return {
      keypoint,
      direction,
      soundType,
      soundValue,
      parameter
    };
  }
  
  // Save custom preset
  public saveCustomPreset(name: string, description: string, mappings: SoundMapping[]): void {
    // In a real application, this would save to localStorage or a database
    // For now, we'll just log it
    console.log('Custom preset saved:', {
      name,
      description,
      instrument: 'custom',
      mappings
    });
  }
}