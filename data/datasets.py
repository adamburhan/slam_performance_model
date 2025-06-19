import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision import transforms

def get_datasets(config):
    root_dir = config['dataset']['root_dir']
    dataset_dict = config['dataset']['dataset_dict']
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    train_dataset = MultiDataset(
        root_dir=root_dir,
        dataset_dict=dataset_dict['train'],
        context_len=config['dataset']['context_len'],
        horizon=config['dataset']['horizon'],
        normalize=False,
        transform=train_transforms
    )
    
    val_dataset = MultiDataset(
        root_dir=root_dir,
        dataset_dict=dataset_dict['val'],
        context_len=config['dataset']['context_len'],
        horizon=config['dataset']['horizon'],
        normalize=False,
        transform=val_transforms
    )
    
    return train_dataset, val_dataset


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
    pass