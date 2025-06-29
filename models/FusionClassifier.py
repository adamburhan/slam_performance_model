import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, weights_path, freeze=True):
        super().__init__()
        resnet = models.resnet18()
        resnet.load_state_dict(torch.load(weights_path, weights_only=True))
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.squeeze(-1).squeeze(-1)
        return features  
    
    
class TemporalImageEncoder(nn.Module):
    def __init__(self, weights_path, context_len, freeze=True):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            # Reduce temporal dimension by 2
            nn.Conv3d(16, 16, kernel_size=(3, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            # Reduce remaining temporal dimension
            nn.Conv3d(16, 3, kernel_size=(context_len-2, 1, 1)),
        )
        resnet = models.resnet18()
        resnet.load_state_dict(torch.load(weights_path, weights_only=True))
        # remove final classif layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
    def forward(self, x): # x.shape: [B, C, 3, H, W]
        x = x.permute(0, 2, 1, 3, 4)  # Change to [B, 3, C, H, W]
        x = self.temporal_conv(x)
        x = x.squeeze(2)  # Remove the temporal dimension
        
        features = self.backbone(x)  # [B, 512, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [B, 512]
        
        return features


class LSTMImageEncoder(nn.Module):
    def __init__(self, weights_path, hidden_dim=512):
        super().__init__()
        resnet = models.resnet18()
        resnet.load_state_dict(torch.load(weights_path, weights_only=True))
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_dim, batch_first=True)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def forward(self, x):  # x.shape: [B, C, 3, H, W]
        B, C, _, H, W = x.shape
        x = x.view(B * C, 3, H, W)  # Flatten
        features = self.backbone(x)  # [B*C, 512, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [B*C, 512]
        features = features.view(B, C, -1)
        output, (h_n, _) = self.lstm(features) 
        return h_n.squeeze(0)  # [B, hidden_dim]
    

class TransformerImageEncoder(nn.Module):
    def __init__(self, weights_path, context_len, num_heads=8):
        super().__init__()
        # Extract features from each frame
        resnet = models.resnet18()
        resnet.load_state_dict(torch.load(weights_path, weights_only=True))
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Process with transformer
        self.position_embedding = nn.Parameter(torch.randn(1, context_len, 512))
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Freeze ResNet weights
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # x shape: [B, C, 3, H, W]
        B, C, _, H, W = x.shape
        x = x.view(B*C, 3, H, W)
        features = self.backbone(x)
        features = features.squeeze(-1).squeeze(-1).view(B, C, -1)  # [B, C, 512]
        
        # Add positional encoding
        features = features + self.position_embedding
        
        # Transformer expects [seq_len, batch, features]
        features = features.permute(1, 0, 2)
        features = self.transformer(features)
        
        # Average over sequence dimension
        features = features.mean(dim=0)  # [B, 512]
        return features

class StatEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x):
        return self.mlp(x)
    
class FusionClassifier(nn.Module):
    def __init__(self, weights_path, num_classes, stat_input_dim, stat_output_dim, img_encoder, context_len):
        super().__init__()
        if img_encoder == 'ImageEncoder':
            self.img_encoder = ImageEncoder(weights_path)
        elif img_encoder == 'TemporalImageEncoder':
            self.img_encoder = TemporalImageEncoder(weights_path, context_len)
        elif img_encoder == 'LSTMImageEncoder':
            self.img_encoder = LSTMImageEncoder(weights_path)
        elif img_encoder == 'TransformerImageEncoder':
            self.img_encoder = TransformerImageEncoder(weights_path, context_len)
        else:
            raise ValueError(f"Unknown image encoder: {img_encoder}")
        
        self.stat_encoder = StatEncoder(stat_input_dim, stat_output_dim)
        self.fuse_dim = 512 + stat_output_dim
        self.shared_mlp = nn.Sequential(
            nn.Linear(self.fuse_dim, self.fuse_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.rot_head = nn.Linear(self.fuse_dim // 2, num_classes)
        self.trans_head = nn.Linear(self.fuse_dim // 2, num_classes)
        
    def forward(self, x):
        """
        imgs -> [B, C, 3, H, W]
        stats -> [B, C, stat_input_dim]
        """
        stats, imgs = x
        
        B, C, _, H, W = imgs.shape
        
        # Different handling based on encoder type
        if isinstance(self.img_encoder, (LSTMImageEncoder, TransformerImageEncoder)):
            # These encoders handle reshaping internally
            img_features = self.img_encoder(imgs)  # [B, 512]
        elif isinstance(self.img_encoder, TemporalImageEncoder):
            # TemporalImageEncoder expects [B, 3, C, H, W]
            img_features = self.img_encoder(imgs)  # [B, 512]
        else:
            # ImageEncoder needs flattened input
            imgs = imgs.view(B * C, 3, H, W)
            img_features = self.img_encoder(imgs)  # [B*C, 512]
            img_features = img_features.view(B, C, -1).mean(dim=1)  # [B, 512]
        
        # Process stats
        stats = stats.view(B * C, -1)
        stat_features = self.stat_encoder(stats)  # [B*C, stat_output_dim]
        stat_features = stat_features.view(B, C, -1).mean(dim=1)  # [B, stat_output_dim]
        
        # Fusion
        fused_features = torch.cat((img_features, stat_features), dim=1)  # [B, fuse_dim]
        
        # Shared processing
        features = self.shared_mlp(fused_features)  # [B, fuse_dim//2]
        
        # Classification heads
        logits_rot = self.rot_head(features)  # [B, num_classes]
        logits_trans = self.trans_head(features)  # [B, num_classes]
        
        return logits_rot, logits_trans