import joblib
from datetime import datetime

class PowerPredictor:
    def __init__(self, model_path):
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            print(f"Model loading failed: {e}")
            self.model = None

    def predict(self, input_data):
        try:
            if self.model:
                return self.model.predict(input_data)[0]
            else:
                raise ValueError("Model is not loaded.")
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.0


class ScalingEngine:
    def __init__(self, upper_thresh=250, lower_thresh=100):
        self.vm_count = 1
        self.scaling_log = []
        self.upper_thresh = upper_thresh
        self.lower_thresh = lower_thresh

    def decide_scaling(self, current_power, predicted_power):
        timestamp = datetime.now().strftime("%H:%M:%S")
        if predicted_power > self.upper_thresh:
            self.vm_count += 1
            action = "Scale UP"
        elif predicted_power < self.lower_thresh and self.vm_count > 1:
            self.vm_count -= 1
            action = "Scale DOWN"
        else:
            action = "No Change"
        self.scaling_log.append([timestamp, current_power, predicted_power, action])
        return action, self.vm_count, self.scaling_log[-10:]
