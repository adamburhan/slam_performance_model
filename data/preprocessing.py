"""
Module for data preprocessing tasks in the SLAM performance model project.

Key Functions:
---------------
1. discretize_and_save_to_csv:
    - This function processes the raw RPE data (rotation and translation errors) from text files 
      for both training and test sets. It calculates quantiles using only the training set to avoid 
      data leakage and applies the same quantile bins to the test set.
    - It outputs CSV files with the following structure:
        rotation_error, translation_error, rotation_quantile, translation_quantile

Usage:
--------
This module is meant to be used as part of the data preprocessing pipeline before loading data 
into the PyTorch DataLoader for training and testing the SLAM performance model.
"""

import pandas as pd
import os
from PIL import Image
import shutil
from pathlib import Path

def discretize_and_save_to_csv(train_files, test_files, output_train_files, output_test_files, n_quantiles=4):
    """
    Discretizes rotation and translation errors for both training and test sets into quantiles,
    and includes the frame ID in the final CSV. Outputs one CSV per sequence.

    Args:
        train_files (list): List of training text files with RPE errors.
        test_files (list): List of testing text files with RPE errors.
        output_train_files (list): List of CSV filenames to save for each training sequence.
        output_test_files (list): List of CSV filenames to save for each test sequence.
        n_quantiles (int): Number of quantiles to create for rotation and translation errors.

    Returns:
        None. Outputs are saved as separate CSV files per sequence.
    """
    
    # Step 1: Load all training data into a single DataFrame for quantile calculation
    train_data = []
    for file in train_files:
        with open(file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                frame_id = int(parts[0])  # last Frame number
                rotation_error = float(parts[1])
                translation_error = float(parts[2])
                train_data.append([frame_id, rotation_error, translation_error])
    
    df_train = pd.DataFrame(train_data, columns=['frame_id', 'rotation_error', 'translation_error'])

    # Step 2: Calculate quantiles based on all training data
    df_train['rotation_quantile'], rotation_bins = pd.qcut(df_train['rotation_error'], q=n_quantiles, labels=False, retbins=True, duplicates='drop')
    df_train['translation_quantile'], translation_bins = pd.qcut(df_train['translation_error'], q=n_quantiles, labels=False, retbins=True, duplicates='drop')

    # Step 3: Now save the training data to separate CSV files (one per sequence)
    for file, output_file in zip(train_files, output_train_files):
        sequence_data = []
        with open(file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                frame_id = int(parts[0])  # Frame number
                rotation_error = float(parts[1])
                translation_error = float(parts[2])
                sequence_data.append([frame_id, rotation_error, translation_error])
        
        df_sequence = pd.DataFrame(sequence_data, columns=['frame_id', 'rotation_error', 'translation_error'])
        df_sequence['rotation_quantile'] = pd.cut(df_sequence['rotation_error'], bins=rotation_bins, labels=False, include_lowest=True)
        df_sequence['translation_quantile'] = pd.cut(df_sequence['translation_error'], bins=translation_bins, labels=False, include_lowest=True)

        # Save the CSV for this sequence
        df_sequence.to_csv(output_file, index=False)

    # Step 4: Apply the same quantile bins to test sequences and save to separate CSV files
    for test_file, output_file in zip(test_files, output_test_files):
        test_data = []
        with open(test_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                frame_id = int(parts[0])
                rotation_error = float(parts[1])
                translation_error = float(parts[2])
                test_data.append([frame_id, rotation_error, translation_error])

        df_test = pd.DataFrame(test_data, columns=['frame_id', 'rotation_error', 'translation_error'])
        df_test['rotation_quantile'] = pd.cut(df_test['rotation_error'], bins=rotation_bins, labels=False, include_lowest=True)
        df_test['translation_quantile'] = pd.cut(df_test['translation_error'], bins=translation_bins, labels=False, include_lowest=True)

        # Save the CSV for this test sequence
        df_test.to_csv(output_file, index=False)

def flip_images(input_dir, output_dir):
    """
    Flip all images in the input directory and save to the output directory, preserving the directory structure.

    Args:
    - input_dir (str): Path to the original dataset directory.
    - output_dir (str): Path to the directory where flipped images will be saved.
    """
    # Walk through all the directories and subdirectories in the input directory
    for root, _, files in os.walk(input_dir):
        # Check if there are any image files in the current directory
        if files:
            # Create a corresponding directory in the output directory
            relative_path = os.path.relpath(root, input_dir)  # Get relative path (e.g., '00/')
            target_dir = os.path.join(output_dir, relative_path)

            # Create the target directory if it does not exist
            os.makedirs(target_dir, exist_ok=True)

            # Loop through each file in the current directory
            for file in files:
                if file.endswith(".png"):  # Ensure that the file is a PNG image
                    input_path = os.path.join(root, file)
                    output_path = os.path.join(target_dir, file)

                    # Open, flip, and save the image
                    with Image.open(input_path) as img:
                        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)  # Flip horizontally
                        flipped_img.save(output_path)

                    #print(f"Flipped {input_path} -> {output_path}")
                    
                    
def discretize_rpe(train_paths, test_paths, out_train_paths, out_test_paths, n_quantiles=5):
    # Load all training RPEs
    train_dfs = [pd.read_csv(p) for p in train_paths]
    full_train = pd.concat(train_dfs, ignore_index=True)

    # Compute bin edges from full training data
    full_train['rot_bin'], rot_bins = pd.qcut(full_train['rpe_rot'], q=n_quantiles, labels=False, retbins=True, duplicates='drop')
    full_train['trans_bin'], trans_bins = pd.qcut(full_train['rpe_trans'], q=n_quantiles, labels=False, retbins=True, duplicates='drop')

    # Apply binning and save per sequence
    for in_path, out_path in zip(train_paths, out_train_paths):
        df = pd.read_csv(in_path)
        df['rot_bin'] = pd.cut(df['rpe_rot'], bins=rot_bins, labels=False, include_lowest=True)
        df['trans_bin'] = pd.cut(df['rpe_trans'], bins=trans_bins, labels=False, include_lowest=True)
        df.to_csv(out_path, index=False)

    for in_path, out_path in zip(test_paths, out_test_paths):
        df = pd.read_csv(in_path)
        df['rot_bin'] = pd.cut(df['rpe_rot'], bins=rot_bins, labels=False, include_lowest=True)
        df['trans_bin'] = pd.cut(df['rpe_trans'], bins=trans_bins, labels=False, include_lowest=True)
        df.to_csv(out_path, index=False)
        
if __name__ == "__main__":
    train_paths = [
        Path("/media/adam/T91/ood_slam_data/datasets/euroc/MH_01_easy/rpe_labels.csv"),
        Path("/media/adam/T91/ood_slam_data/datasets/euroc/MH_03_medium/rpe_labels.csv"),
        Path("/media/adam/T91/ood_slam_data/datasets/euroc/MH_04_difficult/rpe_labels.csv"),
        Path("/media/adam/T91/ood_slam_data/datasets/euroc/MH_05_difficult/rpe_labels.csv"),
        Path("/media/adam/T91/ood_slam_data/datasets/euroc/V1_01_easy/rpe_labels.csv"),
        Path("/media/adam/T91/ood_slam_data/datasets/euroc/V1_03_difficult/rpe_labels.csv"),
        Path("/media/adam/T91/ood_slam_data/datasets/euroc/V2_02_medium/rpe_labels.csv"),
        Path("/media/adam/T91/ood_slam_data/datasets/euroc/V2_01_easy/rpe_labels.csv"),
        Path("/media/adam/T91/ood_slam_data/datasets/euroc/V2_03_difficult/rpe_labels.csv")
    ]
    test_paths = [
        Path("/media/adam/T91/ood_slam_data/datasets/euroc/MH_02_easy/rpe_labels.csv"),
        Path("/media/adam/T91/ood_slam_data/datasets/euroc/V1_02_medium/rpe_labels.csv")
    ]

    out_train_paths = [
        Path("/media/adam/T91/ood_slam_data/datasets/euroc/MH_01_easy/rpe_labels_binned.csv"),
        Path("/media/adam/T91/ood_slam_data/datasets/euroc/MH_03_medium/rpe_labels_binned.csv"),
        Path("/media/adam/T91/ood_slam_data/datasets/euroc/MH_04_difficult/rpe_labels_binned.csv"),
        Path("/media/adam/T91/ood_slam_data/datasets/euroc/MH_05_difficult/rpe_labels_binned.csv"),
        Path("/media/adam/T91/ood_slam_data/datasets/euroc/V1_01_easy/rpe_labels_binned.csv"),
        Path("/media/adam/T91/ood_slam_data/datasets/euroc/V1_03_difficult/rpe_labels_binned.csv"),
        Path("/media/adam/T91/ood_slam_data/datasets/euroc/V2_02_medium/rpe_labels_binned.csv"),
        Path("/media/adam/T91/ood_slam_data/datasets/euroc/V2_01_easy/rpe_labels_binned.csv"),
        Path("/media/adam/T91/ood_slam_data/datasets/euroc/V2_03_difficult/rpe_labels_binned.csv")
    ]
    out_test_paths = [
        Path("/media/adam/T91/ood_slam_data/datasets/euroc/MH_02_easy/rpe_labels_binned.csv"),
        Path("/media/adam/T91/ood_slam_data/datasets/euroc/V1_02_medium/rpe_labels_binned.csv")
    ]

    discretize_rpe(train_paths, test_paths, out_train_paths, out_test_paths, n_quantiles=5)
