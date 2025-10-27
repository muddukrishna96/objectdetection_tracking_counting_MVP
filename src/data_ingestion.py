"""
===========================
 DATA INGESTION INSTRUCTION
===========================

You can customize this file based on your dataset source, structure, and format.
"""

# ----------------------------------------
#  Import Required Libraries
# ----------------------------------------
# Import only the libraries you need for data ingestion.

# Example:
import os
import pandas as pd
from roboflow import Roboflow
import pandas as pd
from pathlib import Path
# You may also use libraries like numpy, requests, boto3, etc., based on your data source.

os.environ.pop("HF_API_KEY", None)
os.environ.pop("HF_LLM_API_URL", None)

# Explicitly force Roboflow to use its correct base API endpoint
os.environ["ROBOFLOW_API_BASE"] = "https://api.roboflow.com"

# ----------------------------------------
# Step 1: Load the Dataset
# ----------------------------------------

def load_data( workspace: str, project_name: str, version_number: int, dataset_format: str = "yolov8-obb") -> str:
    """
    Downloads a dataset from Roboflow and saves it in the current working directory.
    Please set your Roboflow API key in environment variables (ROBOFLOW_API_KEY) before running this script.

    Parameters:
        api_key (str): Roboflow API key.
        workspace (str): Roboflow workspace name.
        project_name (str): Project name in the workspace.
        version_number (int): Version number of the dataset.
        dataset_format (str): Format to download (e.g., 'yolov8-obb', 'coco', 'voc', etc.).

    Returns:
        str: Path to the downloaded dataset directory.
    """
    print("Initializing Roboflow connection...")

    user_api_key = os.getenv("ROBOFLOW_API_KEY")
    if not user_api_key:
        raise ValueError("Roboflow API key not found in environment variables")

    rf = Roboflow(api_key=user_api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_number)
    
    print(f"Downloading dataset '{project_name}' (version {version_number}) in format '{dataset_format}'...")
    dataset = version.download(dataset_format)
    
    dataset_dir = os.path.abspath(dataset.location)
    print(f"✅ Dataset successfully downloaded to: {dataset_dir}")
    
    return dataset_dir



# ----------------------------------------
#  Step 2: Preprocess the Data
# ----------------------------------------


def preprocess_data(dataset_dir: str) -> pd.DataFrame:
    """
    Run structured validation tests on a YOLO-style dataset directory.

    Each check acts like a test case. If any test fails, an error is raised.
    When all tests pass, dataset information is printed and saved as CSV.

    Parameters:
        dataset_dir (str): Path to the dataset directory.

    Returns:
        pd.DataFrame: Summary dataframe with image/label counts per split.
    """
    dataset_path = Path(dataset_dir)
    print(f"\n Running dataset validation tests for: {dataset_path}\n")

    # --- TEST 1: Dataset directory existence ---
    if not dataset_path.exists():
        raise FileNotFoundError(f" TEST FAILED: Dataset directory not found at {dataset_dir}")
    print(" TEST 1 PASSED: Dataset directory exists.")

    # --- TEST 2: Required split folders ---
    required_splits = ["train", "test", "valid"]
    for split in required_splits:
        split_path = dataset_path / split
        if not split_path.exists():
            raise FileNotFoundError(f" TEST FAILED: Missing split folder '{split}' in dataset.")
    print(" TEST 2 PASSED: Found all required split folders (train, test, valid).")

    # --- TEST 3: Required subfolders (images, labels) ---
    for split in required_splits:
        for sub in ["images", "labels"]:
            sub_path = dataset_path / split / sub
            if not sub_path.exists():
                raise FileNotFoundError(f" TEST FAILED: Missing subfolder '{sub}' in '{split}' folder.")
    print(" TEST 3 PASSED: All splits contain 'images' and 'labels' subfolders.")

    # --- TEST 4: Matching images and labels ---
    summary_data = []
    total_images, total_labels = 0, 0

    for split in required_splits:
        img_dir = dataset_path / split / "images"
        lbl_dir = dataset_path / split / "labels"

        images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        labels = sorted([f for f in os.listdir(lbl_dir) if f.lower().endswith('.txt')])

        image_basenames = {os.path.splitext(f)[0] for f in images}
        label_basenames = {os.path.splitext(f)[0] for f in labels}

        missing_labels = image_basenames - label_basenames
        missing_images = label_basenames - image_basenames

        if missing_labels:
            raise AssertionError(f" TEST FAILED: Missing label files for {len(missing_labels)} images in '{split}' folder: {list(missing_labels)[:5]}")
        if missing_images:
            raise AssertionError(f" TEST FAILED: Missing image files for {len(missing_images)} labels in '{split}' folder: {list(missing_images)[:5]}")

        summary_data.append({
            "split": split,
            "num_images": len(images),
            "num_labels": len(labels)
        })

        total_images += len(images)
        total_labels += len(labels)

    print(" TEST 4 PASSED: Each image has a matching label and vice versa.")

    # --- TEST 5: data.yaml existence ---
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f" TEST FAILED: data.yaml file not found in {dataset_dir}")
    print(" TEST 5 PASSED: data.yaml file found.")

    # --- All Tests Passed ---
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = dataset_path / "dataset_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    print("\n ALL TEST CASES PASSED SUCCESSFULLY!")
    print(" Data ingestion successful.\n")
    print(f" Dataset directory: {dataset_dir}")
    print(f" Total images: {total_images}")
    print(f" Total labels: {total_labels}")
    print(f" Summary CSV saved at: {summary_csv_path}\n")

    print(" Dataset Summary:")
    print(summary_df.to_string(index=False))

    return summary_df







# ----------------------------------------
#  Step 4: Define Main Function
# ----------------------------------------
def main():
    """
    Main execution function for the data ingestion pipeline.
    """


    try:
        dataset_path = load_data(
                                    
                                    workspace="enter your workspace",
                                    project_name="-project-xi846",
                                    version_number=3,
                                    dataset_format="yolov11")
        
        dataset_summary = preprocess_data(dataset_path)

    except Exception as e:
        print(f" Error during data ingestion: {e}")
        raise


# ----------------------------------------
#  Entry Point
# ----------------------------------------
if __name__ == "__main__":
    main()
