"""Autoregressive Transformer decoder for LaTeX generation."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding1D(nn.Module):
    """Standard 1D sinusoidal positional encoding for sequences."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding.
        
        Args:
            x: Tensor of shape (B, L, D)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerDecoder(nn.Module):
    """
    Autoregressive Transformer decoder for LaTeX token generation.
    
    Uses causal (look-ahead) masking to ensure autoregressive property.
    Cross-attends to encoder output for image-to-text generation.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        pad_id: int = 0,
    ):
        """
        Initialize decoder.
        
        Args:
            vocab_size: Size of token vocabulary
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of decoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            pad_id: Padding token ID for masking
        """
        super().__init__()
        
        self.d_model = d_model
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.embed_scale = math.sqrt(d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding1D(d_model, max_seq_len, dropout)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )
        
        # Output projection
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Tie embedding and output weights
        self.output_proj.weight = self.embedding.weight
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal (look-ahead) mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def _generate_padding_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Generate padding mask from token IDs."""
        return tokens == self.pad_id  # (B, L)
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for teacher forcing training.
        
        Args:
            tgt: Target token IDs (B, L)
            memory: Encoder output (B, S, D)
            tgt_mask: Causal attention mask (L, L)
            tgt_key_padding_mask: Target padding mask (B, L)
            memory_key_padding_mask: Encoder padding mask (B, S)
            
        Returns:
            Logits over vocabulary (B, L, V)
        """
        B, L = tgt.shape
        
        # Generate causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self._generate_causal_mask(L, tgt.device)
        
        # Generate padding mask if not provided
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self._generate_padding_mask(tgt)
        
        # Embed tokens
        x = self.embedding(tgt) * self.embed_scale  # (B, L, D)
        x = self.pos_encoding(x)
        
        # Apply transformer decoder
        x = self.transformer_decoder(
            x,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        
        x = self.norm(x)
        logits = self.output_proj(x)  # (B, L, V)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        memory: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_len: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.
        
        Args:
            memory: Encoder output (B, S, D)
            sos_id: Start-of-sequence token ID
            eos_id: End-of-sequence token ID
            max_len: Maximum generation length
            temperature: Sampling temperature (1.0 = no change)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Generated token IDs (B, L)
        """
        max_len = max_len or self.max_seq_len
        B = memory.size(0)
        device = memory.device
        
        # Initialize with SOS token
        generated = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        for step in range(max_len - 1):
            # Get logits for next token
            logits = self.forward(generated, memory)  # (B, L, V)
            next_logits = logits[:, -1, :]  # (B, V)
            
            # Apply temperature
            if temperature != 1.0 and temperature > 0:
                next_logits = next_logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')
            
            # Sample or argmax
            if temperature > 0:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            
            # Replace finished sequences with padding
            next_token = next_token.masked_fill(finished.unsqueeze(-1), self.pad_id)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Update finished status
            finished = finished | (next_token.squeeze(-1) == eos_id)
            
            if finished.all():
                break
        
        return generated
    
    @torch.no_grad()
    def beam_search(
        self,
        memory: torch.Tensor,
        sos_id: int,
        eos_id: int,
        beam_size: int = 5,
        max_len: Optional[int] = None,
        length_penalty: float = 0.6,
    ) -> torch.Tensor:
        """
        Beam search decoding.
        
        Args:
            memory: Encoder output (B, S, D) - only B=1 supported
            sos_id: Start-of-sequence token ID
            eos_id: End-of-sequence token ID
            beam_size: Number of beams
            max_len: Maximum generation length
            length_penalty: Length penalty (alpha)
            
        Returns:
            Best sequence (1, L)
        """
        max_len = max_len or self.max_seq_len
        device = memory.device
        
        # Expand memory for beam search
        memory = memory.expand(beam_size, -1, -1)  # (beam, S, D)
        
        # Initialize beams
        beams = torch.full((beam_size, 1), sos_id, dtype=torch.long, device=device)
        beam_scores = torch.zeros(beam_size, device=device)
        beam_scores[1:] = float('-inf')  # Only first beam active initially
        
        finished_beams = []
        finished_scores = []
        
        for step in range(max_len - 1):
            # Get logits
            logits = self.forward(beams, memory)  # (beam, L, V)
            next_logits = logits[:, -1, :]  # (beam, V)
            log_probs = F.log_softmax(next_logits, dim=-1)
            
            vocab_size = log_probs.size(-1)
            
            # Calculate scores for all possible next tokens
            scores = beam_scores.unsqueeze(-1) + log_probs  # (beam, V)
            scores = scores.view(-1)  # (beam * V)
            
            # Get top candidates
            top_scores, top_indices = torch.topk(scores, beam_size * 2)
            
            # Convert flat indices to beam and token indices
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
            
            # Build new beams
            new_beams = []
            new_scores = []
            
            for score, beam_idx, token_idx in zip(top_scores, beam_indices, token_indices):
                # Convert to int and keep 2D shape (1, L) for concatenation
                b_idx = beam_idx.item()
                new_seq = torch.cat([beams[b_idx:b_idx+1], token_idx.view(1, 1)], dim=1)
                
                if token_idx.item() == eos_id:
                    # Finished beam
                    seq_len = new_seq.size(1)
                    final_score = score.item() / (seq_len ** length_penalty)
                    finished_beams.append(new_seq)
                    finished_scores.append(final_score)
                else:
                    new_beams.append(new_seq)
                    new_scores.append(score)
                
                if len(new_beams) >= beam_size:
                    break
            
            if not new_beams:
                break
            
            beams = torch.cat(new_beams, dim=0)
            beam_scores = torch.stack(new_scores)
        
        # Select best finished beam or best current beam
        if finished_beams:
            best_idx = max(range(len(finished_scores)), key=lambda i: finished_scores[i])
            return finished_beams[best_idx]
        else:
            return beams[0:1]

