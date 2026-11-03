import time
import random
from .report_generator import generate_diagnostic_report

def run_inference(state, patient_details):
    """
    Simulates loading a PyTorch .pt file and running a 1D-CNN on the CSV waveform.
    """
    # Simulate the Jetson Nano computing the tensor (1.5 seconds)
    time.sleep(1.5)
    
    # Simulated classes
    labels = ["LBL_0 (No DR)", "LBL_1 (Diabetic-DR)", "LBL_2 (Diabetic-NoDR)"]
    prediction = random.choices(labels, weights=[0.5, 0.25, 0.25])[0]
    confidence = round(random.uniform(78.5, 99.2), 2)
    
    # Generate the Clinical Report
    report_path = generate_diagnostic_report(state, patient_details, prediction, confidence)
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "report_path": report_path
    }