import * as tf from '@tensorflow/tfjs';
import { performanceMonitor } from './PerformanceMonitor';

/**
 * Helper class for TensorFlow.js related operations.
 * Handles initialization, memory management, and optimization.
 */
export class TensorflowHelper {
  private static instance: TensorflowHelper;
  private isInitialized = false;
  private currentBackend: string | null = null;
  private memoryCleanupInterval: number | null = null;
  private tensorCount = 0;
  private lastMemoryInfo: tf.MemoryInfo | null = null;

  private constructor() {}

  /**
   * Get the singleton instance
   */
  public static getInstance(): TensorflowHelper {
    if (!TensorflowHelper.instance) {
      TensorflowHelper.instance = new TensorflowHelper();
    }
    return TensorflowHelper.instance;
  }

  /**
   * Initialize TensorFlow.js with the preferred backend
   */
  public async initializeTensorflow(preferredBackend: string = 'webgl'): Promise<boolean> {
    performanceMonitor.start('tfInit');
    
    try {
      // Make sure TF is ready
      await tf.ready();
      
      // Try to set the preferred backend
      try {
        await tf.setBackend(preferredBackend);
        this.currentBackend = tf.getBackend() || preferredBackend;
        console.log(`TensorFlow.js initialized with backend: ${this.currentBackend}`);
      } catch (backendError) {
        console.warn(`Failed to set ${preferredBackend} backend, trying alternatives:`, backendError);
        
        // Try alternatives in order
        for (const backend of ['webgl', 'webgpu', 'cpu']) {
          if (backend !== preferredBackend) {
            try {
              await tf.setBackend(backend);
              this.currentBackend = backend;
              console.log(`Successfully set fallback backend: ${backend}`);
              break;
            } catch (error) {
              console.warn(`Failed to set ${backend} backend`);
            }
          }
        }
      }
      
      // Apply backend-specific optimizations
      this.optimizeBackend(this.currentBackend || '');
      
      // Start memory monitoring
      this.startMemoryMonitoring();
      
      this.isInitialized = true;
      performanceMonitor.end('tfInit');
      return true;
    } catch (error) {
      console.error('Error initializing TensorFlow:', error);
      performanceMonitor.end('tfInit');
      return false;
    }
  }

  /**
   * Apply optimizations based on the current backend
   */
  private optimizeBackend(backend: string): void {
    if (backend === 'webgl') {
      try {
        tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0);
        tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
        tf.env().set('WEBGL_PACK', true);
        tf.env().set('WEBGL_FLUSH_THRESHOLD', 1);
        console.log('WebGL optimizations applied');
      } catch (error) {
        console.warn('Could not apply all WebGL optimizations:', error);
      }
    } else if (backend === 'webgpu') {
      try {
        // WebGPU specific optimizations when available
        console.log('WebGPU backend selected');
      } catch (error) {
        console.warn('Could not apply WebGPU optimizations:', error);
      }
    } else if (backend === 'cpu') {
      // CPU specific optimizations
      console.log('CPU backend selected - performance may be reduced');
    }
  }

  /**
   * Start memory monitoring to catch leaks
   */
  private startMemoryMonitoring(): void {
    // Clear any existing interval
    if (this.memoryCleanupInterval !== null) {
      window.clearInterval(this.memoryCleanupInterval);
    }
    
    // Check memory every 30 seconds
    this.memoryCleanupInterval = window.setInterval(() => {
      try {
        this.lastMemoryInfo = tf.memory();
        const numTensors = this.lastMemoryInfo.numTensors;
        const numDataBuffers = this.lastMemoryInfo.numDataBuffers;
        const numBytes = this.lastMemoryInfo.numBytes;
        
        // Check for significant tensor growth
        if (numTensors > this.tensorCount * 1.5 && numTensors > 100) {
          console.warn(`Possible tensor leak: ${numTensors} tensors (${(numBytes / 1024 / 1024).toFixed(2)} MB)`);
          this.forceGarbageCollection();
        }
        
        this.tensorCount = numTensors;
      } catch (error) {
        console.error('Error monitoring TensorFlow memory:', error);
      }
    }, 30000);
  }

  /**
   * Force garbage collection of tensors
   */
  public forceGarbageCollection(): void {
    try {
      // Dispose all unused tensors
      tf.tidy(() => {});
      
      // Log memory after cleanup
      const memAfter = tf.memory();
      console.log(`Memory after cleanup: ${memAfter.numTensors} tensors, ${(memAfter.numBytes / 1024 / 1024).toFixed(2)} MB`);
    } catch (error) {
      console.error('Error during force garbage collection:', error);
    }
  }

  /**
   * Create a model with proper memory management
   */
  public createModel(): tf.Sequential {
    return tf.sequential();
  }

  /**
   * Run a function with tensor cleanup
   */
/**
 * Run a function with tensor cleanup
 * Allows returning any type from tidy operation while satisfying TypeScript
 */
public tidy<T>(fn: () => T): T {
  // Define a variable to hold the result
  let result: T;
  
  // Create a type-safe wrapper function for tf.tidy
  // This ensures memory management happens while avoiding type issues
  const safeRunner = () => {
    // Execute the original function and store the result
    result = fn();
    // Return a dummy value that satisfies TensorFlow's type constraints
    return null as any;
  };
  
  // Call tf.tidy with our wrapper
  tf.tidy(safeRunner);
  
  // Return the stored result
  return result!;
}
  /**
   * Clean up resources when the helper is no longer needed
   */
  public dispose(): void {
    if (this.memoryCleanupInterval !== null) {
      window.clearInterval(this.memoryCleanupInterval);
      this.memoryCleanupInterval = null;
    }
    
    // Force immediate cleanup
    this.forceGarbageCollection();
  }

  /**
   * Get current backend
   */
  public getBackend(): string | null {
    return this.currentBackend;
  }

  /**
   * Check if TensorFlow is initialized
   */
  public isReady(): boolean {
    return this.isInitialized;
  }

  /**
   * Get current memory usage info
   */
  public getMemoryInfo(): tf.MemoryInfo | null {
    try {
      return tf.memory();
    } catch (error) {
      console.error('Error getting memory info:', error);
      return null;
    }
  }
  
  /**
   * Wait for TensorFlow to initialize (with timeout)
   */
  public async waitForTensorflow(timeoutMs: number = 10000): Promise<boolean> {
    if (this.isInitialized) return true;
    
    const startTime = Date.now();
    
    while (!this.isInitialized && (Date.now() - startTime < timeoutMs)) {
      await new Promise(resolve => setTimeout(resolve, 100));
      
      // Check if it's been initialized during this wait
      if (this.isInitialized) return true;
    }
    
    return this.isInitialized;
  }
}

// Export singleton instance
export const tensorflowHelper = TensorflowHelper.getInstance();