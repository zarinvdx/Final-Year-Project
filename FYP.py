import pandas as pd
import joblib
from datetime import datetime
from flask import Flask, render_template


class PowerPredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, input_data):
        try:
            return self.model.predict(input_data)[0]
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



app = Flask(__name__)
df = pd.read_csv("cleaned_vmCloud_data.csv")
predictor = PowerPredictor("FYP/predictor.pkl")
scaler = ScalingEngine()

index = 0
cpu_history = []
power_history = []
predicted_history = []
energy_saved_history = []
running_total_energy_saved = 0.0

@app.route('/')
def dashboard():
    global index, cpu_history, power_history, predicted_history, running_total_energy_saved

    try:
        row = df.iloc[index]
        cpu = row['cpu_usage']
        memory = row['memory_usage']
        power = row['power_consumption']
        power_history.append(power)
        energy = round(power / 60, 2)

        efficiency = round((cpu + memory) / energy, 2) if energy != 0 else 0
        baseline_power = 75 * 2.5 + memory * 1.2
        baseline_energy = round(baseline_power / 60, 2)
        energy_saved = max(round(baseline_energy - energy, 2), 0)
        running_total_energy_saved += energy_saved
        co2_saved = round(running_total_energy_saved * 0.475, 2)

        estimated_monthly_saving = round(running_total_energy_saved * 30, 2)
        estimated_cost_saving = round(estimated_monthly_saving * 0.35, 2)

        lag1 = cpu_history[-2] if len(cpu_history) >= 2 else cpu
        lag2 = cpu_history[-3] if len(cpu_history) >= 3 else cpu
        input_data = pd.DataFrame([{
            'cpu_lag1': lag1,
            'cpu_lag2': lag2,
            'mem_lag1': memory - 1,
            'mem_lag2': memory - 2,
            'network_traffic': row['network_traffic'],
            'energy_efficiency': row['energy_efficiency'],
            'hour': pd.to_datetime(row['timestamp']).hour if 'timestamp' in row else row['hour'],
            'day_of_week': pd.to_datetime(row['timestamp']).dayofweek if 'timestamp' in row else row['day_of_week'],
            'cpu_usage': cpu,
            'memory_usage': memory
        }])

        predicted_power = predictor.predict(input_data)
        predicted_history.append(predicted_power)

        action, vm_count, log_tail = scaler.decide_scaling(power, predicted_power)
        alert = {
            "Scale UP": f"High projected power — Scaling UP to {vm_count} VMs",
            "Scale DOWN": f"Low projected power — Scaling DOWN to {vm_count} VMs",
            "No Change": f"All systems normal — {vm_count} VMs running"
        }[action]

        energy_saved_history.append(energy_saved)
        cpu_history.append(cpu)

        index = (index + 1) % len(df)

        return render_template("FYP.html",
                               cpu=round(cpu, 2),
                               memory=round(memory, 2),
                               power=round(power, 2),
                               energy=energy,
                               efficiency=efficiency,
                               predicted_power=round(predicted_power, 2),
                               power_history=power_history[-10:],
                               predicted_history=predicted_history[-10:],
                               alert=alert,
                               vm_count=vm_count,
                               energy_saved=energy_saved,
                               total_energy_saved=round(running_total_energy_saved, 2),
                               co2_saved=co2_saved,
                               energy_saved_history=energy_saved_history[-50:],
                               scaling_log=log_tail,
                               estimated_monthly_saving=estimated_monthly_saving,
                               estimated_cost_saving=estimated_cost_saving,
                               now=datetime.now().strftime("%A, %d %B %Y — %I:%M:%S %p"))

    except Exception as e:
        return f"Dashboard error: {e}"


if __name__ == '__main__':
    app.run(debug=True)
