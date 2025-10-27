"""
====================================
 MODEL TRAINING & EVALUATION TEMPLATE
====================================

This script provides a **template** for model training, evaluation,
and registration using **MLflow** for experiment tracking.

It guides you through:
- Loading engineered features
- Training and evaluating yolo models
- Logging metrics and artifacts with MLflow
- Registering the trained model for versioned deployment
"""
# ----------------------------------------
# Import Required Libraries
# ----------------------------------------
from ultralytics import YOLO
import mlflow
from ultralytics import settings
import yaml
import os
settings.update({'mlflow': True})

# Get directory where this script is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)

# Build full path to parms.yaml
yaml_path = os.path.join(BASE_DIR, "parms.yaml")
print(yaml_path)
with open(yaml_path) as f:
    params = yaml.safe_load(f)


# ----------------------------------------
# Step 1: Initialize MLflow Tracking
# ----------------------------------------
def setup_mlflow_tracking(uri: str, experiment_name: str):
    """
    Configure MLflow tracking server and experiment.
    """
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow Tracking URI set to: {uri}")
    print(f"Experiment: {experiment_name}")


# ----------------------------------------
# Step 2: Main Workflow model training and model logging to mlflow
# ----------------------------------------
def main():
    """
    Main YOLO training pipeline with MLflow integration.
    """
    try:
        # --- Setup MLflow ---
        setup_mlflow_tracking(
            uri=params['mlflow_tracking_uri'],  # Local MLflow tracking server 
            experiment_name=params['experiment_name']
        )

        # --- Train YOLO Model ---
        with mlflow.start_run():
        # Load and train your YOLOv8 model
            print("Logging metrics and model to MLflow...")
            print(params['model_path'])

            model = YOLO(params['model_path'])
            results = model.train(
                data=params['data_yaml_path'],
                epochs=params['epochs'],
                device=params['device'],
                batch=params['batch'],
                imgsz=params['imgsz'],
                workers=params['workers']
            )

            # After training
            model_path = results.save_dir / "weights" / "best.pt"
            print(model_path)


            # Reload a clean model (safe to pickle)
            clean_model = YOLO(str(model_path)).model
            print("clean model reloded")

            mlflow.pytorch.log_model(clean_model, "model")
            print("model logged in mlflow")
  

            # End the run
            mlflow.end_run()


    except Exception as e:
        print(f"Error during YOLO model training: {e}")
        raise


# ----------------------------------------
# Entry Point
# ----------------------------------------
if __name__ == "__main__":
    main()



