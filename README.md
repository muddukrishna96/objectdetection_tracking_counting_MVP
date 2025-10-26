#  Factory Counting System MVP

A real-time computer vision system for counting chocolates and bottles on conveyor belts using YOLO object detection and tracking.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Configuration](#configuration)
- [Results](#results)
- [Troubleshooting](#troubleshooting)

##  Overview

This project implements an automated counting system for factory conveyor belt. Using YOLO11 object detection and tracking, currently the system counts chocolates and bottles as they cross a user-defined line in real-time.in future carious other classes will be added .Trained model is not added in this repo . One can use their own trained model change the model paths in pharms.yaml file .

### Key Capabilities

- **Real-time counting**: Track and count objects crossing a defined line
- **Bi-directional tracking**: Separate counts for IN and OUT directions
- **Multi-class detection**: Chocolates and bottles
- **Multi-camera support**: Process multiple camera feeds simultaneously
- **Visual feedback**: Neon-style bounding boxes with color-coded states
- **MLflow integration**: Track training experiments and model versions

##  Features

- 🎯 **Accurate Detection**: YOLO11 model trained on factory conveyor data
- 📊 **Class-specific Counting**: Tracks counts per object type (chocolate, bottle)
- 🎨 **Neon Visual Effects**: Modern corner-box visualization with glow effects
- 🎥 **Flexible Input**: Support for video files, RTSP streams, and webcams
- 📈 **Experiment Tracking**: MLflow integration for model versioning
- 🔄 **Persistent Tracking**: Object IDs maintained across frames
- ⚡ **Real-time Processing**: Efficient multi-threaded architecture

## 📁 Project Structure

```
factory_chocolat_counting/
│
├── src/
│   ├── data_ingestion.py              # Download datasets from Roboflow
│   ├── model_building.py              # Train YOLO model with MLflow
│   ├── postprocessing_bisunesslogic.py           # Single-camera counting
│   └── postprocessing_bisunesslogic_multicameraprocessing.py  # Multi-camera counting
│
├── parms.yaml                         # Training configuration
├── requriments.txt                    # Python dependencies
```

## 🔧 Requirements

- Python 3.8.11
- CUDA-capable GPU (recommended)

## 📦 Installation

1. **Clone the repository**
```bash
git clone [repository_url]
cd factory_chocolat_counting
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash


# install from requirements file
pip install -r requriments.txt
```

## 🚀 Usage

### Single Camera Counting

For single webcam, RTSP stream, or video file:

```bash
python src/postprocessing_bisunesslogic.py
```

**Interactive setup:**
1. Choose input source (video file or camera)
2. Select counting line: Click 2 points to define the line
3. Watch real-time counting with visual feedback

**Controls:**
- `q`: Quit
- `ENTER`: Confirm line selection
- `ESC`: Exit during line selection

### Multi-Camera Counting

For multiple simultaneous camera feeds:

```bash
python src/postprocessing_bisunesslogic_multicameraprocessing.py
```

**Setup:**
1. Enter camera indices (e.g., `0 1` for two cameras)
2. Draw counting line for each camera
3. System processes all feeds in parallel

### Counting Logic

The system uses line-crossing detection:
- **Red bounding box**: Object detected, not crossed yet
- **Yellow flash**: Object just crossed the line (4 frames)
- **Green bounding box**: Object already counted
- Counts displayed: `[CLASS_NAME] IN: X  OUT: Y`

**Line-crossing algorithm**: Detects which side of the line an object's center point is on and tracks direction changes.

## 🔬 Model Training

### Train Your Own Model

1. **Configure parameters** in `parms.yaml`:
```yaml
model_path: yolo11n.pt
data_yaml_path: My-First-Project-3/data.yaml
epochs: 100
batch: 4
imgsz: 640
device: 0  # GPU device
experiment_name: "choclate_identification_yolo_11"
mlflow_tracking_uri: "http://12.0.0.1:9000"
```

2. **Set up MLflow tracking server**:
```bash
mlflow server 
```

3. **Run training**:
```bash
python src/model_building.py
```

Trained models are automatically logged to MLflow with metrics and artifacts.

4. **View results in MLflow UI**:
```bash
# MLflow UI should be accessible at http://localhost:5000
# View experiments, compare runs, and download models
```

## ⚙️ Configuration

### Training Parameters (`parms.yaml`)

- `model_path`: Pre-trained YOLO model (e.g., `yolo11n.pt`)
- `data_yaml_path`: Dataset configuration file
- `epochs`: Number of training epochs (default: 100)
- `batch`: Batch size (default: 4)
- `imgsz`: Image size (default: 640)
- `device`: GPU/CPU device (`0` for GPU, `cpu` for CPU)
- `experiment_name`: MLflow experiment name
- `mlflow_tracking_uri`: MLflow server URL

### Detection Parameters (in scripts)

- **Confidence threshold**: 0.7 (hardcoded)
- **Classes**: [0, 1] (chocolate=0, bottle=1)
- **Tracking**: Persistent tracking enabled (`persist=True`)
- **Flash duration**: 4 frames after crossing

## 📊 Results

### Model Performance

- **Classes**: Chocolate, Bottle
- **Dataset v3**: 780 train / 75 valid / 37 test images
- **Dataset v2**: 570 train / 55 valid / 27 test images
- **Best Model**: `runs/detect/train5/weights/best.pt`
- **Format**: YOLO8/OBB format

### Counting Features

- **Bi-directional counting**: Accurate IN/OUT detection using line-crossing algorithm
- **Multi-threaded**: Efficient processing of multiple camera feeds
- **Visual feedback**: Real-time neon-style corner boxes
- **Frame persistence**: Objects tracked across frames with unique IDs
- **Per-class tracking**: Separate counts for each object type

## 🎥 Visual Feedback

The system displays:
- **Neon corner boxes**: Modern corner-only visualization with glow effects
- **Center tracking**: Cyan circle at object center
- **Class labels**: Object type and confidence score
- **Counting line**: Red line showing detection boundary
- **Count displays**: Real-time IN/OUT counts per class

Color coding:
- 🔴 Red: Newly detected object
- 🟡 Yellow: Just crossed (flash effect)
- 🟢 Green: Already counted

## sample outputs use case 1 : cloclate factory counting 

![](outputs/counting_output-ezgif.com-video-to-gif-converter.gif)

## case 2 : bottle factory counting 

![](outputs/counting_output_bottle-ezgif.com-video-to-gif-converter.gif)

## 🔍 Dataset Information

### Dataset Version 3 
- **Classes**: 2 (bottle, chocolate)
- **Total images**: 892
- **Format**: YOLO11
- **Source**: Roboflow factory conveyor belt footage


 datasets validated with matching image/label pairs.
 the dataset can be access via this url : https://universe.roboflow.com/monitorshrimps/industrial_dataset-xzhwj


## 📝 License

See LICENSE file for details.

## 🐛 Troubleshooting

**Issue**: Camera not opening
- Check camera permissions
- Verify camera index is correct
- Try different indices (0, 1, 2...)

**Issue**: Model not found
- Verify `best.pt` exists in `runs/detect/train5/weights/`
- Update path in scripts if moved

**Issue**: Poor detection accuracy
- Adjust confidence threshold in scripts
- Retrain with more epochs
- Use more diverse training data

**Issue**: Multi-camera setup fails
- Ensure camera indices are correct
- Check if cameras are already in use
- Verify sufficient USB bandwidth

---

**Built with** **using Ultralytics YOLO, OpenCV, and MLflow**

