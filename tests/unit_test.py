import unittest
import pandas as pd
from unittest.mock import MagicMock
from fyp_components import PowerPredictor, ScalingEngine


class TestPowerPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = PowerPredictor(model_path="FYP/predictor.pkl")
        self.predictor.model = MagicMock()

    def test_predict_valid_input(self):
        dummy_input = pd.DataFrame([[1]*10], columns=[
            'cpu_lag1', 'cpu_lag2', 'mem_lag1', 'mem_lag2',
            'network_traffic', 'energy_efficiency', 'hour', 'day_of_week',
            'cpu_usage', 'memory_usage'
        ])
        self.predictor.model.predict.return_value = [123.45]
        prediction = self.predictor.predict(dummy_input)
        self.assertEqual(prediction, 123.45)

    def test_predict_failure(self):
        self.predictor.model.predict.side_effect = Exception("Failure")
        prediction = self.predictor.predict(pd.DataFrame())
        self.assertEqual(prediction, 0.0)


class TestScalingEngine(unittest.TestCase):
    def setUp(self):
        self.scaler = ScalingEngine(upper_thresh=200, lower_thresh=100)

    def test_scale_up(self):
        action, vm_count, log = self.scaler.decide_scaling(180, 250)
        self.assertEqual(action, "Scale UP")
        self.assertEqual(vm_count, 2)

    def test_scale_down(self):
        self.scaler.vm_count = 2
        action, vm_count, log = self.scaler.decide_scaling(180, 80)
        self.assertEqual(action, "Scale DOWN")
        self.assertEqual(vm_count, 1)

    def test_no_change(self):
        action, vm_count, log = self.scaler.decide_scaling(150, 150)
        self.assertEqual(action, "No Change")
        self.assertEqual(vm_count, 1)

    def test_scaling_log_limit(self):
        for _ in range(15):
            self.scaler.decide_scaling(100, 300)
        _, _, log = self.scaler.decide_scaling(100, 300)
        self.assertLessEqual(len(log), 10)


if __name__ == '__main__':
    unittest.main()