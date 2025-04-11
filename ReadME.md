# 🚗 CARLA Sensor Simulation to Labeled Dataset Workflow

## 🔧 Setup

Install dependencies:

```bash
pip install -r requirements.txt


## 1. Run the CARLA Simulator

Open a terminal and launch CARLA:

```bash
./CarlaUE4.sh
```

Make sure it’s running before moving forward.

---

## 2. Run Your Data Collection Script

This script captures:
- 📸 Camera images  
- 🧲 LiDAR (optional)  
- 📍 GPS, IMU  
- 📦 Actor 3D bounding boxes  
- 🎥 Camera transform

Run the script:

```bash
python sensor_capture.py
```

Let it run for a bit (~100–200 frames), then stop it with \`Ctrl+C\`.

✅ This will populate the \`output/\` directory with:

```
output/
  camera/
  bboxes/
  camera_transforms/
  imu.csv
  gps.csv
```

---

## 3. Run the Dataset Export Script

This script:
- Projects 3D boxes into the 2D image frame
- Annotates each image
- Saves JSON label files
- Builds a dataset index

Run it with:

```bash
python export_dataset.py
```

🟢 Output will appear in: \`output/final_dataset/\`

```
output/final_dataset/
  images/
    000001.png
    ...
  labels/
    000001.json
    ...
  dataset_index.json
```

---

## ✅ Result

You now have a **fully labeled image dataset** from your CARLA simulation that can be used to:
- Train object detection models
- Visualize perception data
- Evaluate scenario-specific model behavior
`;