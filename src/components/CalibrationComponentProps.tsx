import { MLIntentionDetector } from '../utils/MLIntentionDetector';

interface CalibrationComponentProps {
  detector: MLIntentionDetector;
  onCalibrationComplete: (userId: string) => void;
}

export default CalibrationComponentProps;