/**
 * Performance monitoring utility to track execution times
 * and identify bottlenecks in the application.
 */
export class PerformanceMonitor {
    private static instance: PerformanceMonitor;
    private startTimes: Record<string, number> = {};
    private durations: Record<string, number[]> = {};
    private maxSamples = 100;
    private enabled = false;
    private warningThresholds: Record<string, number> = {};
  
    private constructor() {}
  
    /**
     * Get the singleton instance
     */
    public static getInstance(): PerformanceMonitor {
      if (!PerformanceMonitor.instance) {
        PerformanceMonitor.instance = new PerformanceMonitor();
      }
      return PerformanceMonitor.instance;
    }
  
    /**
     * Enable or disable performance monitoring
     */
    public setEnabled(enabled: boolean): void {
      this.enabled = enabled;
    }
  
    /**
     * Start timing a section of code
     */
    public start(label: string): void {
      if (!this.enabled) return;
      this.startTimes[label] = performance.now();
    }
  
    /**
     * End timing a section of code and record the duration
     */
    public end(label: string): number | null {
      if (!this.enabled) return null;
      
      const startTime = this.startTimes[label];
      if (startTime === undefined) {
        console.warn(`PerformanceMonitor: No start time found for label "${label}"`);
        return null;
      }
  
      const duration = performance.now() - startTime;
      
      // Initialize array if it doesn't exist
      if (!this.durations[label]) {
        this.durations[label] = [];
      }
      
      // Add duration to the array
      this.durations[label].push(duration);
      
      // Trim array if it exceeds maxSamples
      if (this.durations[label].length > this.maxSamples) {
        this.durations[label].shift();
      }
      
      // Check if duration exceeds warning threshold
      if (this.warningThresholds[label] && duration > this.warningThresholds[label]) {
        console.warn(`Performance warning: "${label}" took ${duration.toFixed(2)}ms, which exceeds threshold of ${this.warningThresholds[label]}ms`);
      }
      
      return duration;
    }
  
    /**
     * Set a warning threshold for a specific section
     */
    public setWarningThreshold(label: string, thresholdMs: number): void {
      this.warningThresholds[label] = thresholdMs;
    }
  
    /**
     * Get average duration for a labeled section
     */
    public getAverage(label: string): number | null {
      if (!this.durations[label] || this.durations[label].length === 0) {
        return null;
      }
      
      const sum = this.durations[label].reduce((acc, val) => acc + val, 0);
      return sum / this.durations[label].length;
    }
  
    /**
     * Get all performance statistics
     */
    public getStats(): Record<string, { average: number; min: number; max: number; samples: number }> {
      const stats: Record<string, { average: number; min: number; max: number; samples: number }> = {};
      
      Object.keys(this.durations).forEach(label => {
        const samples = this.durations[label];
        if (samples.length === 0) return;
        
        const sum = samples.reduce((acc, val) => acc + val, 0);
        const min = Math.min(...samples);
        const max = Math.max(...samples);
        
        stats[label] = {
          average: sum / samples.length,
          min,
          max,
          samples: samples.length
        };
      });
      
      return stats;
    }
  
    /**
     * Clear all recorded durations
     */
    public reset(): void {
      this.startTimes = {};
      this.durations = {};
    }
    
    /**
     * Measure the execution time of a function
     */
    public measure<T>(label: string, fn: () => T): T {
      this.start(label);
      const result = fn();
      this.end(label);
      return result;
    }
    
    /**
     * Create an async function wrapper that measures execution time
     */
    public async measureAsync<T>(label: string, fn: () => Promise<T>): Promise<T> {
      this.start(label);
      try {
        const result = await fn();
        this.end(label);
        return result;
      } catch (error) {
        this.end(label);
        throw error;
      }
    }
  
    /**
     * Log performance summary to console
     */
    public logSummary(): void {
      console.table(this.getStats());
    }
  }
  
  // Export singleton instance
  export const performanceMonitor = PerformanceMonitor.getInstance();