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

# Define compression levels
COMPRESSION_LEVELS = [
    'high_quality',           # Best quality (95%, no subsampling)
    'high_quality_subsampled', # Balanced (95%, with subsampling)
    'medium_quality',         # Medium compression (75%, no subsampling)
    'high_compression'        # High compression (60%, with subsampling)
]

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

# Define line styles for compression levels
COMPRESSION_STYLES = {
    'high_quality': '-',              # Solid (Best quality)
    'high_quality_subsampled': '--',  # Dashed (Balanced)
    'medium_quality': ':',            # Dotted (Medium compression)
    'high_compression': '-.'          # Dash-dot (High compression)
}

def load_spectral_data(model_dir, compression_level):
    """Load the spectral data from a model's directory for a specific compression level."""
    base_path = Path('CompressedAnalysis') / model_dir / compression_level
    data_file = base_path / 'data.npz'
    
    if not data_file.exists():
        raise FileNotFoundError(f"Spectral data not found for {model_dir} at {compression_level}")
    
    data = np.load(data_file)
    return data['freq'], data['spectra_mean']

def plot_compressed_spectra(models_dict, title_suffix, output_filename):
    plt.figure(figsize=(12, 8))
    
    # Plot spectra for each model
    for model_name, model_dir in models_dict.items():
        try:
            # Plot for each compression level
            for compression in COMPRESSION_LEVELS:
                freq, spectra = load_spectral_data(model_dir, compression)
                
                # Create label with compression level and details
                compression_details = {
                    'high_quality': '95% quality, no subsampling',
                    'high_quality_subsampled': '95% quality, with subsampling',
                    'medium_quality': '75% quality, no subsampling',
                    'high_compression': '60% quality, with subsampling'
                }
                label = f"{model_name} ({compression_details[compression]})"
                
                # Plot with different styles for each compression level
                plt.semilogy(freq, spectra, 
                           label=label, 
                           color=COLORS[model_name],
                           linestyle=COMPRESSION_STYLES[compression],
                           linewidth=2)
            
        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")
    
    # Customize the plot
    plt.title(f"{title_suffix}\nAcross JPEG Compression Levels", fontsize=14, pad=20)
    plt.xlabel('Spatial Frequency (cycles/pixel)', fontsize=12)
    plt.ylabel('Power Spectrum', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits to start at 1e-5
    plt.ylim(1e-5, None)
    
    # Set x-axis limits to focus on relevant frequency range
    plt.xlim(1e-3, 0.5)
    
    # Move legend outside the plot
    plt.legend(fontsize=8, bbox_to_anchor=(1.4, 1), loc='upper right')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot with extra space for the legend
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Plot GAN models
    plot_compressed_spectra(GAN_MODELS, 'GAN Generated Images', 'compressed_spectra_gan.png')
    print("GAN spectra plot has been saved as 'compressed_spectra_gan.png'")
    
    # Plot Diffusion models
    plot_compressed_spectra(DIFFUSION_MODELS, 'Diffusion Generated Images', 'compressed_spectra_diffusion.png')
    print("Diffusion spectra plot has been saved as 'compressed_spectra_diffusion.png'")
    
    # Plot Transformer models
    plot_compressed_spectra(TRANSFORMER_MODELS, 'Transformer Generated Images', 'compressed_spectra_transformer.png')
    print("Transformer spectra plot has been saved as 'compressed_spectra_transformer.png'") 