"""Full Image-to-LaTeX model combining encoder and decoder."""

from typing import Optional

import torch
import torch.nn as nn

from .encoder import ImageEncoder
from .decoder import TransformerDecoder


class Img2LaTeX(nn.Module):
    """
    Full Image-to-LaTeX model.
    
    Combines:
    - ImageEncoder: CNN backbone + 2D positional encoding + Transformer encoder
    - TransformerDecoder: Autoregressive decoder with cross-attention
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        backbone: str = "resnet18",
        pretrained_backbone: bool = True,
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
        in_channels: int = 1,
    ):
        """
        Initialize model.
        
        Args:
            vocab_size: Size of token vocabulary
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder transformer layers
            num_decoder_layers: Number of decoder transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            backbone: CNN backbone type
            pretrained_backbone: Use pretrained CNN weights
            pad_id: Padding token ID
            sos_id: Start-of-sequence token ID
            eos_id: End-of-sequence token ID
            in_channels: Number of input image channels
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        
        # Encoder
        self.encoder = ImageEncoder(
            backbone=backbone,
            pretrained=pretrained_backbone,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            in_channels=in_channels,
        )
        
        # Decoder
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
            pad_id=pad_id,
        )
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for training with teacher forcing.
        
        Args:
            images: Input images (B, C, H, W)
            input_ids: Input token IDs (B, L) - typically [SOS, t1, t2, ...]
            
        Returns:
            Logits over vocabulary (B, L, V)
        """
        # Encode images
        memory, _ = self.encoder(images)  # (B, S, D)
        
        # Decode with teacher forcing
        logits = self.decoder(input_ids, memory)  # (B, L, V)
        
        return logits
    
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images only (for caching in generation).
        
        Args:
            images: Input images (B, C, H, W)
            
        Returns:
            Encoder output (B, S, D)
        """
        memory, _ = self.encoder(images)
        return memory
    
    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_len: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        beam_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate LaTeX tokens from images.
        
        Args:
            images: Input images (B, C, H, W)
            max_len: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            beam_size: Beam size (if None, use greedy/sampling)
            
        Returns:
            Generated token IDs (B, L)
        """
        self.eval()
        
        # Encode images
        memory = self.encode(images)
        
        # Generate using decoder
        if beam_size is not None and beam_size > 1:
            # Beam search (only supports batch size 1)
            if images.size(0) > 1:
                raise ValueError("Beam search only supports batch size 1")
            return self.decoder.beam_search(
                memory,
                sos_id=self.sos_id,
                eos_id=self.eos_id,
                beam_size=beam_size,
                max_len=max_len,
            )
        else:
            return self.decoder.generate(
                memory,
                sos_id=self.sos_id,
                eos_id=self.eos_id,
                max_len=max_len,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
    
    def get_loss(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.
        
        Args:
            images: Input images (B, C, H, W)
            input_ids: Input token IDs (B, L)
            target_ids: Target token IDs (B, L)
            lengths: Sequence lengths for masking (optional)
            
        Returns:
            Scalar loss value
        """
        logits = self.forward(images, input_ids)  # (B, L, V)
        
        # Reshape for cross-entropy
        B, L, V = logits.shape
        logits = logits.view(B * L, V)
        target_ids = target_ids.view(B * L)
        
        # Compute loss, ignoring padding
        loss = nn.functional.cross_entropy(
            logits,
            target_ids,
            ignore_index=self.pad_id,
            reduction='mean',
        )
        
        return loss
    
    @classmethod
    def from_config(cls, config: dict, vocab_size: int) -> "Img2LaTeX":
        """
        Create model from configuration dictionary.
        
        Args:
            config: Model configuration
            vocab_size: Vocabulary size
            
        Returns:
            Model instance
        """
        return cls(
            vocab_size=vocab_size,
            d_model=config.get('d_model', 256),
            nhead=config.get('nhead', 8),
            num_encoder_layers=config.get('num_encoder_layers', 4),
            num_decoder_layers=config.get('num_decoder_layers', 4),
            dim_feedforward=config.get('dim_feedforward', 1024),
            dropout=config.get('dropout', 0.1),
            max_seq_len=config.get('max_seq_len', 256),
            backbone=config.get('backbone', 'resnet18'),
            pretrained_backbone=config.get('pretrained_backbone', True),
            pad_id=config.get('pad_id', 0),
            sos_id=config.get('sos_id', 1),
            eos_id=config.get('eos_id', 2),
            in_channels=config.get('in_channels', 1),
        )

