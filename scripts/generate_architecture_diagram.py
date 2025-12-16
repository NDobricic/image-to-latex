#!/usr/bin/env python3
"""Generate architecture diagram for the Img2LaTeX model."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def draw_block(ax, x, y, width, height, label, color, sublabel=None, fontsize=10):
    """Draw a rounded rectangle block with label."""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color, edgecolor='black', linewidth=1.5,
        alpha=0.9
    )
    ax.add_patch(box)
    
    if sublabel:
        ax.text(x, y + 0.15, label, ha='center', va='center', 
                fontsize=fontsize, fontweight='bold')
        ax.text(x, y - 0.15, sublabel, ha='center', va='center', 
                fontsize=fontsize-2, style='italic', color='#444444')
    else:
        ax.text(x, y, label, ha='center', va='center', 
                fontsize=fontsize, fontweight='bold')
    return box

def draw_arrow(ax, start, end, color='black', style='->', linewidth=2):
    """Draw an arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=linewidth))

def draw_stacked_blocks(ax, x, y, n_blocks, width, height, label, color, spacing=0.08):
    """Draw stacked blocks to represent repeated layers."""
    for i in range(n_blocks):
        offset = (n_blocks - 1 - i) * spacing
        alpha = 0.6 + 0.4 * (i / max(1, n_blocks-1))
        box = FancyBboxPatch(
            (x - width/2 + offset*0.3, y - height/2 - offset), width, height,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=color, edgecolor='black', linewidth=1,
            alpha=alpha
        )
        ax.add_patch(box)
    
    ax.text(x, y, label, ha='center', va='center', 
            fontsize=9, fontweight='bold')

def create_architecture_diagram(output_path):
    """Create the full architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(17, 6.5))
    ax.set_xlim(-0.5, 17)
    ax.set_ylim(0.5, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Colors
    input_color = '#E8F4FD'       # Light blue
    cnn_color = '#FFE4B5'         # Light orange
    transformer_color = '#98FB98' # Light green
    output_color = '#DDA0DD'      # Light purple
    pos_enc_color = '#FFB6C1'     # Light pink
    
    # Y positions
    main_y = 3.5
    
    # === INPUT ===
    draw_block(ax, 0.8, main_y, 1.4, 1.6, 'Input\nImage', input_color, fontsize=11)
    ax.text(0.8, main_y - 1.2, '224×224×1', ha='center', fontsize=10, color='gray')
    
    # === CNN ENCODER ===
    draw_arrow(ax, (1.5, main_y), (2.2, main_y))
    
    # ResNet backbone
    draw_block(ax, 3.2, main_y, 1.6, 2.0, 'ResNet-34\nBackbone', cnn_color, fontsize=11)
    ax.text(3.2, main_y - 1.3, '(pretrained)', ha='center', fontsize=9, color='gray', style='italic')
    
    draw_arrow(ax, (4.0, main_y), (4.6, main_y))
    
    # Feature projection
    draw_block(ax, 5.4, main_y, 1.3, 1.3, 'Conv 1×1', cnn_color, sublabel='→ d_model', fontsize=10)
    
    draw_arrow(ax, (6.05, main_y), (6.6, main_y))
    
    # Positional encoding + Flatten combined
    draw_block(ax, 7.4, main_y + 1.2, 1.5, 0.9, '2D Pos Enc', pos_enc_color, fontsize=10)
    draw_block(ax, 7.4, main_y - 1.2, 1.5, 0.9, 'Flatten', input_color, fontsize=10)
    
    # Plus sign and arrows showing combination
    ax.annotate('', xy=(7.4, main_y + 0.4), xytext=(7.4, main_y + 0.75),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(7.4, main_y - 0.4), xytext=(7.4, main_y - 0.75),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(7.4, main_y, '+', ha='center', va='center', fontsize=18, fontweight='bold')
    
    draw_arrow(ax, (8.15, main_y), (8.7, main_y))
    
    # === TRANSFORMER ENCODER ===
    draw_stacked_blocks(ax, 9.6, main_y, 3, 1.5, 1.6, 'Transformer\nEncoder ×4', transformer_color)
    ax.text(9.6, main_y - 1.2, 'Self-Attention', ha='center', fontsize=9, color='gray')
    
    # === MEMORY CONNECTION (more space) ===
    draw_arrow(ax, (10.35, main_y), (11.6, main_y))
    ax.text(11.0, main_y + 0.4, 'Memory', ha='center', fontsize=10, fontweight='bold', color='#2E7D32')
    
    # === DECODER SECTION (shifted right for more space) ===
    decoder_x = 12.6
    
    # Transformer Decoder
    draw_stacked_blocks(ax, decoder_x, main_y, 3, 1.5, 1.6, 'Transformer\nDecoder ×4', transformer_color)
    ax.text(decoder_x, main_y - 1.2, 'Cross-Attention', ha='center', fontsize=9, color='gray')
    
    # Target input (above decoder)
    draw_block(ax, decoder_x, main_y + 2.5, 1.8, 0.9, 'Target (shifted)', input_color, fontsize=10)
    
    # 1D Pos Enc (between target and decoder)
    draw_block(ax, decoder_x, main_y + 1.4, 1.4, 0.7, '+ 1D Pos Enc', pos_enc_color, fontsize=9)
    
    # Arrows in decoder
    draw_arrow(ax, (decoder_x, main_y + 2.05), (decoder_x, main_y + 1.75))
    draw_arrow(ax, (decoder_x, main_y + 1.05), (decoder_x, main_y + 0.85))
    
    # === OUTPUT ===
    draw_arrow(ax, (decoder_x + 0.75, main_y), (decoder_x + 1.4, main_y))
    
    draw_block(ax, decoder_x + 2.2, main_y, 1.4, 1.2, 'Linear +\nSoftmax', output_color, fontsize=10)
    
    draw_arrow(ax, (decoder_x + 2.9, main_y), (decoder_x + 3.4, main_y))
    ax.text(decoder_x + 3.5, main_y, 'LaTeX\nTokens', ha='left', va='center', fontsize=11, fontweight='bold')
    
    # === ANNOTATIONS ===
    # Encoder box (extended bottom padding)
    encoder_box = mpatches.FancyBboxPatch(
        (2.2, main_y - 2.0), 8.3, 4.5,
        boxstyle="round,pad=0.05,rounding_size=0.2",
        facecolor='none', edgecolor='#1976D2', linewidth=2.5,
        linestyle='--', alpha=0.8
    )
    ax.add_patch(encoder_box)
    ax.text(6.2, main_y + 2.8, 'Encoder', ha='center', fontsize=13, fontweight='bold', color='#1976D2')
    
    # Decoder box
    decoder_box = mpatches.FancyBboxPatch(
        (decoder_x - 1.1, main_y - 2.0), 2.2, 5.4,
        boxstyle="round,pad=0.05,rounding_size=0.2",
        facecolor='none', edgecolor='#D32F2F', linewidth=2.5,
        linestyle='--', alpha=0.8
    )
    ax.add_patch(decoder_box)
    ax.text(decoder_x, main_y + 3.7, 'Decoder', ha='center', fontsize=13, fontweight='bold', color='#D32F2F')
    
    # Hyperparameter annotations at top
    ax.text(3.2, main_y + 3.2, 'd_model = 384', ha='center', fontsize=10, 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
    ax.text(6.2, main_y + 3.2, 'd_ff = 1536', ha='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
    ax.text(9.3, main_y + 3.2, 'heads = 8', ha='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Architecture diagram saved to {output_path}")


if __name__ == "__main__":
    output_path = "report/Plots/architecture.png"
    create_architecture_diagram(output_path)
