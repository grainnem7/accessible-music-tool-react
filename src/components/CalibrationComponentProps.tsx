import { MLIntentionDetector } from '../utils/IntentionDetection/MLIntentionDetector';

interface CalibrationComponentProps {
  detector: MLIntentionDetector;
  onCalibrationComplete: (userId: string) => void;
}

export default CalibrationComponentProps;