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
    if (!this.isInitialized) {
      await Tone.start();
      console.log('Tone.js audio context started');
      this.isInitialized = true;
    }
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
          { keypoint: 'left_wrist', direction: 'down', soundType: 'chord', soundValue: ['F3', 'A3', 'C4'] }
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
          { keypoint: 'left_wrist', direction: 'right', soundType: 'drum', soundValue: 'F2' } // Clap
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
          { keypoint: 'nose', direction: 'right', soundType: 'note', soundValue: 'G2', parameter: 'filter' }
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
          { keypoint: 'left_shoulder', direction: 'up', soundType: 'note', soundValue: 'G4' }
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
    if (!isIntentional || !this.isInitialized) return;
    
    // Find matching mappings
    const matchingMappings = this.mappings.filter(mapping => 
      mapping.keypoint === keypoint && 
      (mapping.direction === direction || mapping.direction === 'any')
    );
    
    matchingMappings.forEach(mapping => {
      // Normalize velocity to 0-1 range for sound parameters
      const normalizedVelocity = Math.min(Math.max(velocity / 50, 0), 1);
      
      switch (mapping.soundType) {
        case 'note':
          this.playNote(mapping.soundValue as string, normalizedVelocity, mapping.parameter, keypoint);
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
    });
  }
  
  // Play a single note
  private playNote(note: string, velocity: number, parameter?: string, keypoint?: string): void {
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
      this.synth.triggerAttackRelease(note, "8n", Tone.now(), velocity);
    }
  }
  
  // Play a chord
  private playChord(notes: string[], velocity: number): void {
    this.synth.triggerAttackRelease(notes, "4n", Tone.now(), velocity);
  }
  
  // Play a drum sound
  private playDrum(note: string, velocity: number): void {
    this.drums.triggerAttackRelease(note, "8n", Tone.now(), velocity);
  }
  
  // Play a special effect
  private playEffect(effect: string, intensity: number): void {
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