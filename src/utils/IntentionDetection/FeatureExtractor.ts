import * as poseDetection from '@tensorflow-models/pose-detection';
import { 
  MovementFeatures, 
  TRACKED_KEYPOINTS 
} from './DetectionTypes';
import { performanceMonitor } from '../Helpers/PerformanceMonitor';

/**
 * Feature Extractor - Extracts movement features from pose data
 * Designed to work both in the main thread and in a web worker
 */
export class FeatureExtractor {
  // Movement tracking state
  private movementStart = new Map<string, {x: number, y: number, time: number}>();
  private isInMovement = new Map<string, boolean>();
  private movementPatterns = new Map<string, number[]>();
  
  // Feature cache to avoid recalculating
  private featureCache = new Map<string, { timestamp: number, features: MovementFeatures }>();
  private cacheValidityPeriod = 50; // ms
  
  // Constants
  private readonly MIN_SIGNIFICANT_MOVEMENT = 2;
  private readonly MOVEMENT_START_THRESHOLD = 5;
  private readonly MOVEMENT_END_THRESHOLD = 3;
  
  constructor() {}
  
  /**
   * Extract movement features from a sequence of poses for a specific keypoint
   */
  public extractFeatures(
    poseHistory: Array<{poses: poseDetection.Pose[], timestamp: number}>, 
    keypointName: string
  ): MovementFeatures | null {
    // Use performance monitoring to track extraction time
    performanceMonitor.start('extractFeatures');
    
    try {
      // Check for valid input
      if (poseHistory.length < 10) {
        return null;
      }
      
      // Check cache first for very recent calculations
      const cacheKey = `${keypointName}-${poseHistory[poseHistory.length-1].timestamp}`;
      const cachedResult = this.featureCache.get(cacheKey);
      
      if (cachedResult && 
          (Date.now() - cachedResult.timestamp < this.cacheValidityPeriod)) {
        return cachedResult.features;
      }
      
      // Use more of the history for better analysis
      const recentHistory = poseHistory.slice(-15);
      const oldestPose = recentHistory[0].poses[0];
      const newestPose = recentHistory[recentHistory.length - 1].poses[0];
      
      const oldKeypoint = this.findKeypointByName(oldestPose, keypointName);
      const newKeypoint = this.findKeypointByName(newestPose, keypointName);
      
      if (!oldKeypoint || !newKeypoint || 
          oldKeypoint.x === undefined || newKeypoint.x === undefined) {
        return null;
      }
      
      // Calculate velocity
      const timeElapsed = recentHistory[recentHistory.length - 1].timestamp - recentHistory[0].timestamp;
      const velocityX = (newKeypoint.x - oldKeypoint.x) / (timeElapsed / 1000);
      const velocityY = (newKeypoint.y - oldKeypoint.y) / (timeElapsed / 1000);
      
      // Calculate magnitude of movement
      const dx = newKeypoint.x - oldKeypoint.x;
      const dy = newKeypoint.y - oldKeypoint.y;
      const magnitudeOfMovement = Math.sqrt(dx * dx + dy * dy);
      
      // Calculate acceleration
      let acceleration = 0;
      if (recentHistory.length >= 3) {
        const midIndex = Math.floor(recentHistory.length / 2);
        const midPose = recentHistory[midIndex].poses[0];
        const midKeypoint = this.findKeypointByName(midPose, keypointName);
        
        if (midKeypoint && midKeypoint.x !== undefined) {
          const midTime = recentHistory[midIndex].timestamp;
          
          const velX1 = (midKeypoint.x - oldKeypoint.x) / ((midTime - recentHistory[0].timestamp) / 1000);
          const velY1 = (midKeypoint.y - oldKeypoint.y) / ((midTime - recentHistory[0].timestamp) / 1000);
          
          const velX2 = (newKeypoint.x - midKeypoint.x) / ((recentHistory[recentHistory.length - 1].timestamp - midTime) / 1000);
          const velY2 = (newKeypoint.y - midKeypoint.y) / ((recentHistory[recentHistory.length - 1].timestamp - midTime) / 1000);
          
          const accX = (velX2 - velX1) / (timeElapsed / 1000);
          const accY = (velY2 - velY1) / (timeElapsed / 1000);
          
          acceleration = Math.sqrt(accX * accX + accY * accY);
        }
      }
      
      // Detect if a movement is starting
      const isMovementStarting = magnitudeOfMovement > this.MOVEMENT_START_THRESHOLD && 
                              !this.isInMovement.get(keypointName);
      
      // If movement is starting, record start position and time
      if (isMovementStarting) {
        this.movementStart.set(keypointName, {
          x: oldKeypoint.x,
          y: oldKeypoint.y,
          time: recentHistory[0].timestamp
        });
        this.isInMovement.set(keypointName, true);
      }
      
      // Detect if movement is ending
      const isMovementEnding = magnitudeOfMovement < this.MOVEMENT_END_THRESHOLD && 
                            this.isInMovement.get(keypointName);
      
      // If movement is ending, calculate duration
      let durationOfMovement = 0;
      let isReversing = false;
      
      if (isMovementEnding) {
        const start = this.movementStart.get(keypointName);
        if (start) {
          durationOfMovement = (recentHistory[recentHistory.length - 1].timestamp - start.time) / 1000;
          
          // Check if the movement reversed direction
          const totalDx = newKeypoint.x - start.x;
          const totalDy = newKeypoint.y - start.y;
          
          // If the movement ended close to where it started, it might be a reversing movement
          const totalDistance = Math.sqrt(totalDx * totalDx + totalDy * totalDy);
          if (totalDistance < magnitudeOfMovement * 0.5) {
            isReversing = true;
          }
        }
        
        this.isInMovement.set(keypointName, false);
      } else if (this.isInMovement.get(keypointName)) {
        // Movement is still ongoing
        const start = this.movementStart.get(keypointName);
        if (start) {
          durationOfMovement = (recentHistory[recentHistory.length - 1].timestamp - start.time) / 1000;
        }
      }
      
      // Calculate jitter (deviation from straight line)
      const jitter = this.calculateJitter(recentHistory, keypointName);
      
      // Calculate direction
      let direction = 'none';
      if (Math.abs(dx) > Math.abs(dy)) {
        direction = dx > 0 ? 'right' : 'left';
      } else {
        direction = dy > 0 ? 'down' : 'up';
      }
      
      // Determine if movement is smooth
      const isSmooth = jitter < 10; // Threshold for smoothness
      
      // Calculate frequency of direction changes
      const frequencyOfMovement = this.calculateFrequencyOfMovement(recentHistory, keypointName);
      
      // Calculate steadiness (inverse of jitter variance)
      const steadiness = this.calculateSteadiness(recentHistory, keypointName);
      
      // Calculate pattern recognition score
      const patternScore = this.calculatePatternScore(recentHistory, keypointName);
      
      // Calculate continuity
      const continuity = this.calculateContinuity(recentHistory, keypointName);
      
      // Create feature object
      const features: MovementFeatures = {
        keypoint: keypointName,
        velocityX,
        velocityY,
        acceleration,
        jitter,
        direction,
        isSmooth,
        timestamp: Date.now(),
        magnitudeOfMovement,
        durationOfMovement,
        isReversing,
        frequencyOfMovement,
        steadiness,
        patternScore,
        continuity
      };
      
      // Cache the result
      this.featureCache.set(cacheKey, {
        timestamp: Date.now(),
        features
      });
      
      // Clean old cache entries
      this.cleanCache();
      
      performanceMonitor.end('extractFeatures');
      return features;
    } catch (error) {
      console.error('Error extracting features:', error);
      performanceMonitor.end('extractFeatures');
      return null;
    }
  }
  
/**
 * Clear old entries from the feature cache
 */
private cleanCache(): void {
    const now = Date.now();
    // Instead of direct iteration, convert to array first
    Array.from(this.featureCache.entries()).forEach(([key, value]) => {
      if (now - value.timestamp > 1000) { // 1 second expiration
        this.featureCache.delete(key);
      }
    });
    
    // Limit cache size
    if (this.featureCache.size > 100) {
      // Delete oldest entries
      const entries = Array.from(this.featureCache.entries());
      entries.sort((a, b) => a[1].timestamp - b[1].timestamp);
      
      // Remove oldest half
      for (let i = 0; i < entries.length / 2; i++) {
        this.featureCache.delete(entries[i][0]);
      }
    }
  }
  
  /**
   * Helper to find a keypoint by name
   */
  private findKeypointByName(pose: poseDetection.Pose, name: string): poseDetection.Keypoint | undefined {
    return pose.keypoints.find(kp => kp.name === name);
  }
  
  /**
   * Calculate jitter as deviation from a straight line
   */
  private calculateJitter(
    recentHistory: Array<{poses: poseDetection.Pose[], timestamp: number}>,
    keypointName: string
  ): number {
    // Extract x,y coordinates for this keypoint over recent frames
    const coordinates: {x: number, y: number, timestamp: number}[] = [];
    
    recentHistory.forEach(history => {
      const keypoint = this.findKeypointByName(history.poses[0], keypointName);
      if (keypoint && keypoint.x !== undefined && keypoint.y !== undefined) {
        coordinates.push({
          x: keypoint.x, 
          y: keypoint.y,
          timestamp: history.timestamp
        });
      }
    });
    
    if (coordinates.length < 5) return 0;
    
    // Calculate average movement vector
    const startPoint = coordinates[0];
    const endPoint = coordinates[coordinates.length - 1];
    const vectorX = endPoint.x - startPoint.x;
    const vectorY = endPoint.y - startPoint.y;
    const vectorLength = Math.sqrt(vectorX * vectorX + vectorY * vectorY);
    
    // If barely any movement, return 0
    if (vectorLength < this.MIN_SIGNIFICANT_MOVEMENT) return 0;
    
    // Calculate average deviation from the straight line between start and end
    let totalDeviation = 0;
    
    for (let i = 1; i < coordinates.length - 1; i++) {
      const point = coordinates[i];
      
      // Calculate progress along the line (0 to 1)
      const timeProgress = (point.timestamp - startPoint.timestamp) / 
                          (endPoint.timestamp - startPoint.timestamp);
      
      // Expected position if movement was perfectly linear
      const expectedX = startPoint.x + (vectorX * timeProgress);
      const expectedY = startPoint.y + (vectorY * timeProgress);
      
      // Calculate deviation from expected position
      const deviationX = point.x - expectedX;
      const deviationY = point.y - expectedY;
      const deviation = Math.sqrt(deviationX * deviationX + deviationY * deviationY);
      
      totalDeviation += deviation;
    }
    
    return totalDeviation / (coordinates.length - 2); // Average deviation
  }
  
  /**
   * Calculate how frequently direction changes (tremors have high frequency)
   */
  private calculateFrequencyOfMovement(
    recentHistory: Array<{poses: poseDetection.Pose[], timestamp: number}>,
    keypointName: string
  ): number {
    // Extract coordinates
    const coordinates: {x: number, y: number, timestamp: number}[] = [];
    
    recentHistory.forEach(history => {
      const keypoint = this.findKeypointByName(history.poses[0], keypointName);
      if (keypoint && keypoint.x !== undefined && keypoint.y !== undefined) {
        coordinates.push({
          x: keypoint.x, 
          y: keypoint.y,
          timestamp: history.timestamp
        });
      }
    });
    
    if (coordinates.length < 5) return 0;
    
    // Detect direction changes
    let directionChanges = 0;
    let prevDx = 0;
    let prevDy = 0;
    
    for (let i = 1; i < coordinates.length; i++) {
      const dx = coordinates[i].x - coordinates[i-1].x;
      const dy = coordinates[i].y - coordinates[i-1].y;
      
      // Check if X direction changed
      if ((dx > 0 && prevDx < 0) || (dx < 0 && prevDx > 0)) {
        directionChanges++;
      }
      
      // Check if Y direction changed
      if ((dy > 0 && prevDy < 0) || (dy < 0 && prevDy > 0)) {
        directionChanges++;
      }
      
      prevDx = dx;
      prevDy = dy;
    }
    
    // Calculate frequency (changes per second)
    const timeSpan = (coordinates[coordinates.length-1].timestamp - coordinates[0].timestamp) / 1000;
    return timeSpan > 0 ? directionChanges / timeSpan : 0;
  }
  
  /**
   * Calculate steadiness (low variance = steady, high = erratic)
   */
  private calculateSteadiness(
    recentHistory: Array<{poses: poseDetection.Pose[], timestamp: number}>,
    keypointName: string
  ): number {
    // Extract velocities
    const velocities: number[] = [];
    
    for (let i = 1; i < recentHistory.length; i++) {
      const prevPose = recentHistory[i-1].poses[0];
      const currPose = recentHistory[i].poses[0];
      
      const prevKeypoint = this.findKeypointByName(prevPose, keypointName);
      const currKeypoint = this.findKeypointByName(currPose, keypointName);
      
      if (prevKeypoint && currKeypoint && 
          prevKeypoint.x !== undefined && prevKeypoint.y !== undefined &&
          currKeypoint.x !== undefined && currKeypoint.y !== undefined) {
        
        const dx = currKeypoint.x - prevKeypoint.x;
        const dy = currKeypoint.y - prevKeypoint.y;
        const velocity = Math.sqrt(dx*dx + dy*dy);
        
        velocities.push(velocity);
      }
    }
    
    if (velocities.length < 3) return 0;
    
    // Calculate variance of velocities
    const mean = velocities.reduce((sum, v) => sum + v, 0) / velocities.length;
    const variance = velocities.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / velocities.length;
    
    // Convert variance to steadiness (inverse relationship)
    // Higher variance means less steady
    return Math.max(0, 1 - Math.min(1, variance / 100));
  }
  
  /**
   * Calculate pattern matching score
   */
  private calculatePatternScore(
    recentHistory: Array<{poses: poseDetection.Pose[], timestamp: number}>,
    keypointName: string
  ): number {
    // Extract movement pattern
    const pattern: number[][] = [];
    
    for (let i = 1; i < recentHistory.length; i++) {
      const prevPose = recentHistory[i-1].poses[0];
      const currPose = recentHistory[i].poses[0];
      
      const prevKeypoint = this.findKeypointByName(prevPose, keypointName);
      const currKeypoint = this.findKeypointByName(currPose, keypointName);
      
      if (prevKeypoint && currKeypoint && 
          prevKeypoint.x !== undefined && prevKeypoint.y !== undefined &&
          currKeypoint.x !== undefined && currKeypoint.y !== undefined) {
        
        const dx = currKeypoint.x - prevKeypoint.x;
        const dy = currKeypoint.y - prevKeypoint.y;
        
        pattern.push([dx, dy]);
      }
    }
    
    if (pattern.length < 5) return 0;
    
    this.movementPatterns.set(keypointName, pattern.reduce((acc, val) => acc.concat(val), []));
    
    // For intentional movements, patterns often have a structure
    // We'll look for repeating patterns or consistent directions
    
    // Check for consistent direction
    let consistentXDirection = true;
    let consistentYDirection = true;
    
    if (pattern.length > 0) {
      const firstX = pattern[0][0];
      const firstY = pattern[0][1];
      
      for (let i = 1; i < pattern.length; i++) {
        if ((pattern[i][0] > 0) !== (firstX > 0)) consistentXDirection = false;
        if ((pattern[i][1] > 0) !== (firstY > 0)) consistentYDirection = false;
      }
    }
    
    // Higher score if direction is consistent
    return (consistentXDirection || consistentYDirection) ? 0.8 : 0.2;
  }
  
  /**
   * Calculate continuity of movement
   */
  private calculateContinuity(
    recentHistory: Array<{poses: poseDetection.Pose[], timestamp: number}>,
    keypointName: string
  ): number {
    // Extract velocities at each timestep
    const velocities: number[] = [];
    
    for (let i = 1; i < recentHistory.length; i++) {
      const prevPose = recentHistory[i-1].poses[0];
      const currPose = recentHistory[i].poses[0];
      const timeDiff = (recentHistory[i].timestamp - recentHistory[i-1].timestamp) / 1000;
      
      const prevKeypoint = this.findKeypointByName(prevPose, keypointName);
      const currKeypoint = this.findKeypointByName(currPose, keypointName);
      
      if (prevKeypoint && currKeypoint && 
          prevKeypoint.x !== undefined && prevKeypoint.y !== undefined &&
          currKeypoint.x !== undefined && currKeypoint.y !== undefined &&
          timeDiff > 0) {
        
        const dx = currKeypoint.x - prevKeypoint.x;
        const dy = currKeypoint.y - prevKeypoint.y;
        const velocity = Math.sqrt(dx*dx + dy*dy) / timeDiff;
        
        velocities.push(velocity);
      }
    }
    
    if (velocities.length < 3) return 0;
    
    // Count how many velocities are close to zero (pauses in movement)
    const pauseThreshold = 5; // Velocity below this is considered a pause
    const pauses = velocities.filter(v => v < pauseThreshold).length;
    
    // Calculate continuity as percentage of non-paused frames
    return Math.max(0, 1 - (pauses / velocities.length));
  }
  
  /**
   * Clear feature extraction state
   */
  public clearState(): void {
    this.movementStart.clear();
    this.isInMovement.clear();
    this.movementPatterns.clear();
    this.featureCache.clear();
  }
}

// Create singleton instance for main thread use
export const featureExtractor = new FeatureExtractor();