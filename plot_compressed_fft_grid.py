import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Define the models and their directories
GAN_MODELS = {
    # Real datasets first
    'COCO (Real)': 'comparative_real_coco_valid',
    'UCID (Real)': 'comparative_real_ucid',
    'ImageNet (Real)': 'comparative_real_imagenet_valid',
    # GAN models
    'BigGAN': 'comparative_biggan_256',
    'ProGAN': 'comparative_progan_lsun',
    'StyleGAN2': 'comparative_stylegan2_ffhq_256x256',
    'StyleGAN3': 'comparative_stylegan3_r_ffhqu_256x256',
    'EG3D': 'comparative_eg3d'
}

DIFFUSION_MODELS = {
    # Real datasets first
    'COCO (Real)': 'comparative_real_coco_valid',
    'UCID (Real)': 'comparative_real_ucid',
    'ImageNet (Real)': 'comparative_real_imagenet_valid',
    # Diffusion models
    'Guided Diffusion': 'comparative_guided-diffusion_class2image_ImageNet',
    'GLIDE': 'comparative_glide_text2img_valid',
    'Latent Diffusion': 'comparative_latent-diffusion_class2image_ImageNet',
    'Stable Diffusion': 'comparative_stable_diffusion_256'
}

TRANSFORMER_MODELS = {
    # Real datasets first
    'COCO (Real)': 'comparative_real_coco_valid',
    'UCID (Real)': 'comparative_real_ucid',
    'ImageNet (Real)': 'comparative_real_imagenet_valid',
    # Transformer models
    'Taming Transformers': 'comparative_taming-transformers_class2image_ImageNet',
    'DALL-E Mini': 'comparative_dalle-mini_valid'
}

# Define compression levels and their directories
COMPRESSION_LEVELS = {
    'high_quality': 'high_quality',  # 95% quality, no subsampling
    'high_quality_subsampled': 'high_quality_subsampled',  # 95% quality, with subsampling
    'medium_quality': 'medium_quality',  # 75% quality, no subsampling
    'high_compression': 'high_compression'  # 60% quality, with subsampling
}

# Define colors and styles
COLORS = {
    # Real datasets in different shades of black/gray
    'COCO (Real)': '#000000',      # Black
    'UCID (Real)': '#404040',      # Dark gray
    'ImageNet (Real)': '#808080',  # Medium gray
    # GAN models in colors
    'BigGAN': '#1f77b4',          # blue
    'ProGAN': '#ff7f0e',          # orange
    'StyleGAN2': '#2ca02c',        # green
    'StyleGAN3': '#d62728',        # red
    'EG3D': '#9467bd',            # purple
    # Diffusion models in different colors
    'Guided Diffusion': '#17becf',  # cyan
    'GLIDE': '#bcbd22',            # yellow-green
    'Latent Diffusion': '#e377c2',  # pink
    'Stable Diffusion': '#7f7f7f',  # gray
    # Transformer models in different colors
    'Taming Transformers': '#8c564b',  # brown
    'DALL-E Mini': '#ff9896'          # light red
}

def load_fft_data(model_dir, compression_level):
    """Load the FFT fingerprint data from a model's compressed directory."""
    base_path = Path('CompressedAnalysis') / model_dir / compression_level
    data_file = base_path / 'data.npz'
    
    if not data_file.exists():
        raise FileNotFoundError(f"FFT data not found for {model_dir} at {compression_level}")
    
    data = np.load(data_file)
    return data['res_fft2_mean']

def plot_fft_grid(models_dict, compression_level, title_suffix, output_filename):
    # Create a figure with subplots for each model
    n_models = len(models_dict)
    n_cols = 3  # Number of columns in the grid
    n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    # Plot FFT fingerprint for each model
    for idx, (model_name, model_dir) in enumerate(models_dict.items()):
        try:
            fft_data = load_fft_data(model_dir, compression_level)
            # Average across color channels if present
            fft_data = np.mean(fft_data, -1) if fft_data.ndim > 2 else fft_data
            
            # Plot the FFT fingerprint
            im = axes[idx].imshow(fft_data.clip(0, 1), 
                                clim=[0, 1], 
                                extent=[-0.5, 0.5, 0.5, -0.5],
                                cmap='viridis')
            axes[idx].set_title(model_name, fontsize=10, pad=15)  # Increased padding for subplot titles
            axes[idx].axis('off')  # Hide axes for cleaner look
            
        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")
            axes[idx].axis('off')  # Hide axes for empty subplots
    
    # Hide any remaining empty subplots
    for idx in range(len(models_dict), len(axes)):
        axes[idx].axis('off')
    
    # Add a main title with more space from the subplots
    compression_details = {
        'high_quality': '95% quality, no subsampling',
        'high_quality_subsampled': '95% quality, with subsampling',
        'medium_quality': '75% quality, no subsampling',
        'high_compression': '60% quality, with subsampling'
    }
    fig.suptitle(f"{title_suffix} - FFT Fingerprints\n({compression_details[compression_level]})", 
                 fontsize=14, y=0.99)
    
    # Adjust layout to prevent overlap, with extra space at the top
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Added rect parameter to reserve space for title
    
    # Save the plot
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Generate plots for each compression level
    for compression_level in COMPRESSION_LEVELS:
        # Plot GAN models
        plot_fft_grid(GAN_MODELS, compression_level, 
                     'GAN Generated Images', 
                     f'compressed_fft_gan_{compression_level}.png')
        print(f"GAN FFT fingerprint plot for {compression_level} has been saved as 'compressed_fft_gan_{compression_level}.png'")
        
        # Plot Diffusion models
        plot_fft_grid(DIFFUSION_MODELS, compression_level, 
                     'Diffusion Generated Images', 
                     f'compressed_fft_diffusion_{compression_level}.png')
        print(f"Diffusion FFT fingerprint plot for {compression_level} has been saved as 'compressed_fft_diffusion_{compression_level}.png'")
        
        # Plot Transformer models
        plot_fft_grid(TRANSFORMER_MODELS, compression_level, 
                     'Transformer Generated Images', 
                     f'compressed_fft_transformer_{compression_level}.png')
        print(f"Transformer FFT fingerprint plot for {compression_level} has been saved as 'compressed_fft_transformer_{compression_level}.png'") 