import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import numpy as np

class KITTIDataset(Dataset):
    def __init__(self, sequence_dirs, flipped_sequence_dirs, label_files, sequence_length=2, transform=None):
        """
        Args:
            sequence_dirs (list): List of directories with images for each KITTI sequence.
            label_files (list): List of CSV files with RPE labels for each sequence.
            sequence_length (int): Number of consecutive frames in a sequence (default is 2).
            transform (callable, optional): Optional transform to be applied to the images.
        """
        self.sequence_dirs = sequence_dirs
        self.flipped_sequence_dirs = flipped_sequence_dirs
        self.sequence_length = sequence_length
        self.transform = transform

        # Load all label files into a list of DataFrames, one per sequence
        self.labels = [pd.read_csv(label_file) for label_file in label_files]

        # Calculate and store the length of each sequence (minus sequence_length + 1)
        self.seq_lengths = [len(label_df) - sequence_length + 1 for label_df in self.labels]

        # Store cumulative lengths for indexing across multiple sequences
        self.cum_lengths = [sum(self.seq_lengths[:i+1]) for i in range(len(self.seq_lengths))]

    def __len__(self):
        # Total length is the sum of all sequence lengths
        return sum(self.seq_lengths)

    def _get_sequence_idx(self, idx):
        # Find which sequence idx falls into
        for seq_num, cum_len in enumerate(self.cum_lengths):
            if idx < cum_len:
                if seq_num == 0:
                    return seq_num, idx
                return seq_num, idx - self.cum_lengths[seq_num - 1]

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sequence to retrieve.
        
        Returns:
            A dictionary containing:
            - 'images': A tensor of shape [sequence_length, 6, H, W] where 6 is for concatenated RGB images.
            - 'rpe': A tensor containing the RPE (rotation_error, translation_error) for the last frame in the sequence.
        """
        # Determine which sequence and the specific frame within that sequence
        seq_num, frame_idx = self._get_sequence_idx(idx)

        # Randomly decide whether to use flipped data or not
        use_flipped = random.choice([True, False])

        base_dir = self.flipped_sequence_dirs if use_flipped else self.sequence_dirs

        frames = []  # To store the sequence of frames
        label_df = self.labels[seq_num]

        # Loop over sequence length to get consecutive frames
        for i in range(self.sequence_length):
            frame_id = str(int(label_df.iloc[frame_idx + i]['frame_id'])).zfill(6)
            seq_dir = base_dir[seq_num]  # Get the image directory for the current sequence

            # Load stereo images (left and right) from image_0 and image_1 directories
            img_left_path = os.path.join(seq_dir, "image_0", f"{frame_id}.png")
            img_right_path = os.path.join(seq_dir, "image_1", f"{frame_id}.png")
            img_left = Image.open(img_left_path).convert('RGB')
            img_right = Image.open(img_right_path).convert('RGB')

            # Apply transformations if any
            if self.transform:
                img_left = self.transform(img_left)
                img_right = self.transform(img_right)

            # Concatenate the left and right images along the channel dimension (6 channels: RGB for each)
            stereo_img = torch.cat((img_left, img_right), dim=0)
            frames.append(stereo_img)

        # Concatenate the frames to create a tensor of shape [sequence_length * channels, H, W]
        sequence = torch.cat(frames, dim=0)

        # Get the RPE (rotation and translation errors) for the last frame in the sequence
        rpe = label_df.iloc[frame_idx + self.sequence_length - 1][['rotation_quantile', 'translation_quantile']].values
        rpe = torch.tensor(rpe, dtype=torch.float32)
        rpe = rpe.type(torch.LongTensor)

        return {'images': sequence, 'rpe': rpe}

class MultiDataset(Dataset):
    def __init__(self, root_dir, dataset_dict, context_len=2, horizon=5,
                 normalize=True, transform=None):
        """_summary_

        Args:
            root_dir (_type_): _description_
            dataset_dict (_type_): _description_
            context_len (int, optional): _description_. Defaults to 2.
            horizon (int, optional): _description_. Defaults to 5.
            normalize (bool, optional): _description_. Defaults to True.
            transform (_type_, optional): _description_. Defaults to None.
        """
        
        self.context_len = context_len
        self.horizon = horizon
        self.normalize = normalize
        self.transform = transform
        self.samples = []
        self.feature_dfs = {}
        self.rpe_dfs = {}
        self.image_dirs = {}
        
        for dataset_name, sequences in dataset_dict.items():
            for seq_name in sequences:
                seq_path = os.path.join(root_dir, dataset_name, seq_name)
                self._load_sequence(dataset_name, seq_name, seq_path)
                
                
    def _load_sequence(self, dataset_name, seq_name, seq_path):
        """"Load and process a single sequence"""
        # load data files
        features_path = os.path.join(seq_path, "vo_features.csv")
        rpe_path = os.path.join(seq_path, "rpe_labels_binned.csv")
        image_dir = os.path.join(seq_path, "images_cam0")
        
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features not found: {features_path}")
        if not os.path.exists(rpe_path):
            raise FileNotFoundError(f"RPE labels not found: {rpe_path}")
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
   

        feature_df = pd.read_csv(features_path)
        rpe_df = pd.read_csv(rpe_path)
        
        self.feature_dfs[(dataset_name, seq_name)] = feature_df
        self.rpe_dfs[(dataset_name, seq_name)] = rpe_df
        self.image_dirs[(dataset_name, seq_name)] = image_dir
        
        for _, rpe_row in rpe_df.iterrows():
            frame_id = int(rpe_row['id'])
            
            # check if there is sufficient context
            if self._has_sufficient_context(feature_df, frame_id):
                self.samples.append({
                    'dataset': dataset_name,
                    'sequence': seq_name,
                    'frame_id': frame_id,
                    'label': rpe_row[['rot_bin', 'trans_bin']].values
                })
                
    def _has_sufficient_context(self, feature_df, frame_id):
        """Check if context frames exist before this frame_id"""
        # Get all frame IDs in features
        feature_frame_ids = set(feature_df['id'].values)
        
        # Generate required context frame IDs
        required_context = range(frame_id - self.context_len, frame_id)
        
        # Check if all required context frames exist
        return all(fid in feature_frame_ids for fid in required_context)
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        dataset = sample['dataset']
        seq = sample['sequence']
        frame_id = sample['frame_id']
        
        # internal slam features
        feature_df = self.feature_dfs[(dataset, seq)]
        image_dir = self.image_dirs[(dataset, seq)]
        
        context_frame_ids = range(frame_id - self.context_len, frame_id)

        # retrieve internal slam features for context
        context_features = []
        for fid in context_frame_ids:
            row = feature_df[feature_df['id'] == fid]
            if len(row) == 0:
                raise ValueError(f"Missing features for frame {fid} in {dataset}/{seq}")
        
            # extract features
            features = row.drop(columns=['id', 'timestamp']).values[0]
            context_features.append(features)
            
        # image context
        context_images = []
        for fid in context_frame_ids:
            img_path = os.path.join(image_dir, f"{fid:06d}.png")
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            context_images.append(img)
            
        # convert to tensors
        features_tensor = torch.tensor(np.array(context_features), dtype=torch.float32)
        images_tensor = torch.stack(context_images, dim=0)  # [context_len, 3, H, W]
        images_tensor = images_tensor.permute(1, 0, 2, 3).reshape(3 * self.context_len, 224, 224)
        label_tensor = torch.tensor(sample['label'], dtype=torch.float32)
        
        return (features_tensor, images_tensor), label_tensor

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),          # Convert images to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
    ])

    # Combine all the training sequences (00–08)
    train_sequence_dirs = [f'/media/adam/T9/ood_slam_data/datasets/kitti/odometry_gray/sequences/{i:02d}' for i in range(9)]
    train_label_files = [f'/media/adam/T9/slam_performance_model/data/errors/discretized/{i:02d}.csv' for i in range(9)]

    # Initialize the dataset
    train_dataset = KITTIDataset(sequence_dirs=train_sequence_dirs, label_files=train_label_files, sequence_length=2, transform=transform)

    # Create a DataLoader for the dataset
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Test loading a batch of data
    for batch in train_loader:
        images = batch['images']  # Tensor of shape [batch_size, sequence_length, 6, H, W]
        rpe = batch['rpe']        # Tensor of shape [batch_size, 2]
        print(f"Images shape: {images.shape}, RPE shape: {rpe.shape}")
        break
