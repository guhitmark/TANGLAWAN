<div align="center">

<img src="vision_capture/web/assets/logo.png" alt="TANGLAWAN Logo" width="140"/>

# TANGLAWAN AI
### *Tanglaw ng Inobasyon, Liwanag sa Bawat Aksyon*

**Binocular Chromatic Pupillometry System for Diabetic Retinopathy Screening**

[![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20Nano%20B01-76b900?logo=nvidia)](https://developer.nvidia.com/embedded/jetson-nano)
[![Python](https://img.shields.io/badge/Python-3.6%2B-blue?logo=python)](https://python.org)
[![Framework](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Model](https://img.shields.io/badge/Inference-ONNX%20Runtime-lightgrey?logo=onnx)](https://onnxruntime.ai)
[![License](https://img.shields.io/badge/License-Research%20Prototype-orange)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Pre--clinical%20Validation-yellow)]()

</div>

---

> **⚠️ Research Prototype — Not a Medical Device.**
> TANGLAWAN AI is a capstone engineering research system under active development. It is not approved for clinical diagnostic use. All testing involving human participants requires prior IRB/REC ethics approval.

---

## What is TANGLAWAN?

TANGLAWAN AI is a wearable, self-contained pupillometry headset that screens for early-stage **Diabetic Retinopathy (DR)** — *before* visible structural damage appears in the eye. It does this by measuring the **Post-Illumination Pupillary Response (PIPR)**, a retinal biomarker driven by intrinsically photosensitive retinal ganglion cells (ipRGCs) that express melanopsin. These cells are selectively vulnerable to early DR.

The entire test takes **under 5 minutes**, requires **no pupil dilation**, and produces an **automated AI diagnostic output** — all offline, on the edge, with zero cloud dependency.

```
Patient wears headset → 300-second chromatic PLR sequence → 
OpenCV pupil tracking → PIPR waveform extraction → 
1D-CNN inference on Jetson GPU → Diagnostic report
```

---

## Key Features

- 🔴🔵 **Dual-channel chromatic stimulation** — 624 nm red (photopic / cone) + 460 nm blue (melanopsin / ipRGC)
- 👁️ **Binocular NIR pupil tracking** — dual OV9281 NOIR global shutter cameras at 30 fps
- 🧠 **Onboard AI inference** — 1D-CNN classifier on Jetson Nano GPU via ONNX Runtime (TensorRT FP16)
- 🌐 **Browser-based clinical dashboard** — React frontend, FastAPI/Uvicorn backend, no internet required
- 🔒 **Offline edge architecture** — all data stays on device; Wi-Fi AP hosted by the Jetson itself
- ⚡ **Battery-powered** — 4-cell 18650 BMS, 5W power mode, untethered clinical operation
- 📋 **Automated PDF diagnostic report** — printable or uploadable to the TANGLAWAN mobile web app
- 🛡️ **IEC 62471 Exempt Group certified** — DOE-LATL validated photobiological safety (624 nm & 460 nm)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TANGLAWAN HEADSET                        │
│                                                             │
│  [OV9281 NOIR L] ──┐                  ┌── [LED Module L]    │
│                    ├── Jetson Nano ────┤                    │
│  [OV9281 NOIR R] ──┘   B01 (5W)       └── [LED Module R]    │
│                         │                                   │
│                    [18650 BMS]                              │
└─────────────────────────┼───────────────────────────────────┘
                          │ Wi-Fi AP (SSID: TANGLAWAN)
                          │
              ┌───────────▼──────────────┐
              │   Clinician's Browser    │
              │  http://10.42.0.1:8000   │
              │  (any device, no app)    │
              └──────────────────────────┘
```

### LED Stimulus Channels

| Channel | Wavelength | Target Photoreceptor | Role |
|---------|-----------|----------------------|------|
| IR (×4/eye) | 940 nm | — | Continuous pupil illumination (invisible) |
| Red | 624 nm | L-cones (photopic) | PLR baseline stimulus |
| Blue | 460 nm | ipRGCs (melanopsin) | PIPR induction stimulus |

### Diagnostic Sequence (300 s total)

```
0–10s    Red Baseline   │  11s      Red Pulse (1s)
12–60s   Red PIPR       │  61–120s  Dark Readaptation
121–130s Blue Baseline  │  131s     Blue Pulse (1s)
132–180s Blue PIPR      ←── 1D-CNN classifies this window
```

---

## Pupil Tracking Performance (RITnet-Lite · TensorRT FP16)

| Metric | PyTorch (FP32) | TensorRT (FP16) |
|--------|---------------|-----------------|
| Dice Score | 0.966 | 0.966 |
| Pixel Accuracy | 0.999 | 0.999 |
| Diameter MAE | 0.127 mm | 0.127 mm |
| Inference Latency | ~80 ms | **~24.8 ms** |
| Throughput (single eye) | 12.5 FPS | **40.4 FPS** |
| System RAM | 2,424 MB | **1,156 MB** |
| GPU Load | 84.8% | **56.0%** |
| Power Draw | 6,348 mW | **5,895 mW** |

*Evaluated on 108 frames. Bias: −0.006 mm · LoA: [−0.327, +0.316] mm · CCC: 0.969*

---

## Repository Structure

```
vision_capture/
├── backend/
│   ├── config.py              # GPIO pins, camera indices, LED timing constants
│   └── stimulus_led.py        # PWM LED sequencing and MOSFET control
├── dataset/
│   ├── annotated/             # Exported pupil diameter CSVs
│   └── recordings/            # Raw binocular .mp4 recordings per patient
├── hardware/
│   ├── 3d_models/             # STL files for all PETG enclosure components
│   └── schematics/            # KiCad project + Gerber files for PCB fabrication
├── models/
│   ├── 1d_cnn_classifier.pt   # PyTorch checkpoint (retraining reference)
│   └── 1d_cnn_classifier.onnx # ONNX graph for Jetson Runtime
├── pupil_annotator/
│   ├── static/                # React dashboard frontend
│   └── server.py              # FastAPI server — routes, WebSocket, recording
├── analysis/
│   └── plr_analysis.py        # Offline PLR/PIPR extraction and statistics
├── run.py                     # Entry point — starts Uvicorn, initializes hardware
└── led_spectro_test.py        # Standalone LED diagnostic / bench test
```

---

## Hardware Requirements

| Component | Specification |
|-----------|--------------|
| Compute | NVIDIA Jetson Nano B01 Developer Kit |
| Storage | 128 GB microSD (Class 10, UHS-I) |
| Cameras (×2) | OV9281 **NOIR** global shutter USB — 30 fps, UVC |
| Power | 4-cell 18650 UPS Battery Module B (BMS, ~5V/3A out) |
| IR LEDs (×8) | 940 nm, 5mm through-hole |
| Red LED (×2) | 624 nm, 5mm through-hole |
| Blue LED (×2) | 460 nm, 5mm through-hole |
| LED Driver | 2N7002 N-channel MOSFET (×3 per module) |
| PCB | Custom KiCad design — fabricated at DOST AMC |
| Enclosure | 3D-printed **black PETG** (7 STL files) |
| OS | JetPack **4.6.4** (Ubuntu 18.04, CUDA 10.2) |

> ⚠️ **Use the NOIR variant of the OV9281.** Standard modules include an IR cut filter that blocks 940 nm — the pupil will be invisible.

---

## Quickstart

### 1. Flash JetPack 4.6.4
Format with **SD Card Formatter**, write image with **BalenaEtcher**. Do not use JetPack 5.x (Jetson Nano not supported).

### 2. First Boot (Headless via Micro-USB)
Place the **J48 jumper** on the carrier board. Connect Micro-USB to your PC. SSH via PuTTY:
```
Host: 192.168.55.1   Port: 22   User: tanglawan
```
Complete the setup wizard:
```
Username : tanglawan
Password : 630nmredled
Hostname : tanglawan
```

### 3. Set 5W Power Mode
```bash
sudo nvpmodel -m 1 && sudo nvpmodel -q
# Expected: NV Power Mode: MODE_5W
```

### 4. Install Dependencies
```bash
sudo apt update && sudo apt install -y \
  python3-pip python3-dev libopencv-dev python3-opencv \
  v4l-utils i2c-tools git

pip3 install --break-system-packages \
  fastapi "uvicorn[standard]" numpy opencv-python-headless \
  onnxruntime psutil pillow Jetson.GPIO

sudo groupadd -f -r gpio && sudo usermod -aG gpio tanglawan
```

### 5. Clone & Launch
```bash
cd ~/Desktop
git clone https://github.com/<your-org>/vision_capture.git
cd vision_capture
bash start.sh
```
Then connect to the **TANGLAWAN** Wi-Fi and open **http://10.42.0.1:8000** in any browser.

---

## Clinical Operation

### System Boot
1. Locate the square access port on the **left side** of the headset enclosure.
2. Slide the power switch **left** to power ON, **right** to power OFF.

### Running a Diagnostic Session
1. **Patient prep** — Seat the patient in a dimly lit room for ≥ 5 minutes prior.
2. **Fit the headset** — Position over the eyes, pull the rear support to the back of the head, tighten the locking dial and velcro strap until snug and light-tight.
3. **Launch dashboard** — SSH into the Jetson, run `bash start.sh`, then Ctrl+click the link.
4. **Configure trial** — Enter the 6-digit Patient ID, select **UNKNOWN (Run Inference)** from the diagnosis dropdown, and fill in the Clinical Intake Details form.
5. **Start recording** — Press **S** or click **Start**. The 300-second chromatic sequence runs automatically.
6. **Generate report** — Click **Generate Report** when prompted. The AI result (LBL_0 / LBL_1 / LBL_2) and confidence appear within ~60 seconds. Click **View Report & Close**.

### Safe Shutdown
```bash
sudo shutdown now
```
Wait for the Jetson's internal green LED to turn off, then slide the power switch right.

### Hygiene Protocol
- Wipe face cushion and head straps with **70% isopropyl alcohol** wipes between patients.
- Let air-dry fully before the next patient.
- **Never spray liquid** directly onto lenses or inside the enclosure.
- Clean lenses with a dry microfiber optical cloth only.

> ⚡ **Electrical Safety** — Never operate the device while it is plugged into a wall charger. All diagnostic sessions must run on internal battery power only.

---

## Wi-Fi Configuration

To connect the Jetson Nano to a new Wi-Fi network:
```bash
# List available networks
nmcli d wifi list

# Connect
sudo nmcli d wifi connect "WIFI_NAME" password "WIFI_PASSWORD"

# Get new IP address
hostname -I
```

---

## Diagnostic Report Output

The system generates an automated PDF report containing:
- Physician and patient information
- Clinical metrics (HbA1c, BP, general status)
- **1D-CNN inference result** (LBL_0 = No DR · LBL_1 = Diabetic-DR · LBL_2 = Diabetic-NoDR)
- Model confidence score
- PLR waveform plots — Red stimulus (0–60 s) and Blue stimulus (240–300 s), both eyes

Reports can be printed to PDF or uploaded to the TANGLAWAN mobile web app for remote patient access.

---

## Model Training

The 1D-CNN is trained externally (GPU workstation / Google Colab) and deployed as an ONNX model.

```python
# Export to ONNX after training
torch.onnx.export(model, dummy_input, '1d_cnn_classifier.onnx',
                  opset_version=11,
                  input_names=['pipr_waveform'],
                  output_names=['class_logits'])
```

Copy the `.onnx` file to `models/` on the Jetson. The runtime automatically selects the CUDA Execution Provider when the GPU build of ONNX Runtime is installed.

---

## Safety & Compliance

| Standard | Status |
|----------|--------|
| IEC 62471 (Photobiological Safety) | ✅ Exempt Group — DOE-LATL verified (624 nm & 460 nm) |
| RA 10173 (Philippine Data Privacy Act) | ✅ Offline processing; patient data anonymized on-device |
| Electrical Isolation | ✅ Battery-only operation during all patient sessions |

---

## Project Status

| Milestone | Status |
|-----------|--------|
| Hardware fabrication & optical verification | ✅ Complete |
| RITnet-Lite TensorRT pupil segmentation | ✅ Complete |
| React dashboard + FastAPI backend | ✅ Complete |
| IEC 62471 DOE safety testing | ✅ Complete |
| Spatial calibration (sub-mm accuracy) | ✅ Complete |
| 1D-CNN architecture defined | ✅ Complete |
| 1D-CNN training dataset collection | 🔄 In Progress (PELI pilot) |
| IRB / REC ethics approval | 🔄 In Progress |
| Clinical validation (sensitivity / specificity) | ⏳ Pending |

---

## Roadmap

- [ ] Hardware power switches for improved clinical ergonomics
- [ ] Expanded chromatic stimuli (green channel) for glaucoma / AMD screening
- [ ] CNN-LSTM or Time-Series Transformer for improved waveform classification
- [ ] Multi-center clinical validation
- [ ] Longitudinal DR progression monitoring
- [ ] Next-generation compute: Jetson Orin Nano

---

## Team

Developed by fourth-year Bachelor of Science in Electronics Engineering from **Technological University of the Philippines**.

Clinical validation pilot: **Peregrine Eye and Laser Institute (PELI)**
PCB fabrication: **DOST Advanced Manufacturing Center**
LED safety testing: **DOE Lighting and Technology Laboratory (DOE-LATL)**

---

## License

This repository is released for **academic and research reference only**. The system is a research prototype and must not be used as a standalone clinical diagnostic tool without appropriate regulatory clearance.

---

<div align="center">

*"Tanglaw ng Inobasyon, Liwanag sa Bawat Aksyon"*
**Light of Innovation, Brightness in Every Action**

</div>
