"""Image encoder: CNN backbone + 2D positional encoding + Transformer encoder."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models


class PositionalEncoding2D(nn.Module):
    """
    2D sinusoidal positional encoding for image feature maps.
    
    Creates separate encodings for height and width dimensions,
    then concatenates them to form the full positional encoding.
    """
    
    def __init__(self, d_model: int, max_h: int = 64, max_w: int = 64, dropout: float = 0.1):
        """
        Initialize 2D positional encoding.
        
        Args:
            d_model: Model dimension (must be divisible by 4)
            max_h: Maximum height
            max_w: Maximum width
            dropout: Dropout rate
        """
        super().__init__()
        
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D PE"
        
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # Create position encodings
        pe_h = self._create_1d_encoding(max_h, d_model // 2)
        pe_w = self._create_1d_encoding(max_w, d_model // 2)
        
        # Register as buffers (not parameters)
        self.register_buffer('pe_h', pe_h)
        self.register_buffer('pe_w', pe_w)
    
    def _create_1d_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create 1D sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add 2D positional encoding to feature map.
        
        Args:
            x: Feature tensor of shape (B, H, W, D) or (B, H*W, D)
            
        Returns:
            Tensor with positional encoding added
        """
        if x.dim() == 3:
            # Assume square-ish feature map
            B, L, D = x.shape
            H = W = int(math.sqrt(L))
            x = x.view(B, H, W, D)
        
        B, H, W, D = x.shape
        
        # Get positional encodings for current size
        pe_h = self.pe_h[:H, :]  # (H, D/2)
        pe_w = self.pe_w[:W, :]  # (W, D/2)
        
        # Expand to full grid
        pe_h = pe_h.unsqueeze(1).expand(-1, W, -1)  # (H, W, D/2)
        pe_w = pe_w.unsqueeze(0).expand(H, -1, -1)  # (H, W, D/2)
        
        # Concatenate h and w encodings
        pe = torch.cat([pe_h, pe_w], dim=-1)  # (H, W, D)
        pe = pe.unsqueeze(0)  # (1, H, W, D)
        
        x = x + pe
        return self.dropout(x)


class CNNBackbone(nn.Module):
    """
    CNN backbone for extracting visual features.
    
    Uses a pretrained ResNet with the final FC layer removed.
    """
    
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        output_channels: int = 256,
        in_channels: int = 1,
    ):
        """
        Initialize CNN backbone.
        
        Args:
            backbone: ResNet variant ("resnet18", "resnet34", "resnet50")
            pretrained: Whether to use ImageNet pretrained weights
            output_channels: Output feature dimension
            in_channels: Number of input channels (1 for grayscale)
        """
        super().__init__()
        
        # Load pretrained ResNet
        if backbone == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            resnet_channels = 512
        elif backbone == "resnet34":
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            resnet_channels = 512
        elif backbone == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            resnet_channels = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Modify first conv layer for grayscale input
        if in_channels != 3:
            old_conv = resnet.conv1
            resnet.conv1 = nn.Conv2d(
                in_channels, 64,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            # Initialize with mean of original weights
            if pretrained:
                with torch.no_grad():
                    resnet.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        
        # Extract layers up to layer4 (before avgpool and fc)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        
        # Project to desired output dimension
        self.proj = nn.Conv2d(resnet_channels, output_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from image.
        
        Args:
            x: Input image tensor (B, C, H, W)
            
        Returns:
            Feature map (B, H', W', D) where H', W' are reduced spatial dims
        """
        x = self.features(x)  # (B, C, H', W')
        x = self.proj(x)      # (B, D, H', W')
        x = x.permute(0, 2, 3, 1)  # (B, H', W', D)
        return x


class ImageEncoder(nn.Module):
    """
    Full image encoder: CNN backbone + 2D PE + Transformer encoder.
    
    Processes images into a sequence of contextualized feature vectors
    that can be attended to by the decoder.
    """
    
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        in_channels: int = 1,
    ):
        """
        Initialize image encoder.
        
        Args:
            backbone: CNN backbone type
            pretrained: Whether to use pretrained CNN weights
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            in_channels: Number of input channels
        """
        super().__init__()
        
        self.d_model = d_model
        
        # CNN backbone
        self.cnn = CNNBackbone(
            backbone=backbone,
            pretrained=pretrained,
            output_channels=d_model,
            in_channels=in_channels,
        )
        
        # 2D positional encoding
        self.pos_encoding = PositionalEncoding2D(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        images: torch.Tensor,
        return_2d: bool = False,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        Encode images to feature sequences.
        
        Args:
            images: Input images (B, C, H, W)
            return_2d: Whether to return 2D feature map instead of flattened
            
        Returns:
            features: Encoded features (B, L, D) where L = H' * W'
            spatial_size: (H', W') spatial dimensions of feature map
        """
        # CNN feature extraction
        features = self.cnn(images)  # (B, H', W', D)
        B, H, W, D = features.shape
        
        # Add 2D positional encoding
        features = self.pos_encoding(features)  # (B, H', W', D)
        
        if return_2d:
            return features, (H, W)
        
        # Flatten spatial dimensions for transformer
        features = features.view(B, H * W, D)  # (B, L, D)
        
        # Apply transformer encoder
        features = self.transformer_encoder(features)  # (B, L, D)
        features = self.norm(features)
        
        return features, (H, W)

