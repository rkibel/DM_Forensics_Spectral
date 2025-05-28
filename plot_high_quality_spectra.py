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

# Define line styles
LINE_STYLES = {
    # Real datasets with solid lines
    'COCO (Real)': '-',           # Solid
    'UCID (Real)': '-',           # Solid
    'ImageNet (Real)': '-',       # Solid
    # GAN models with dashed lines
    'BigGAN': '--',              # Dashed
    'ProGAN': '--',              # Dashed
    'StyleGAN2': '--',           # Dashed
    'StyleGAN3': '--',           # Dashed
    'EG3D': '--',                # Dashed
    # Diffusion models with dotted lines
    'Guided Diffusion': ':',      # Dotted
    'GLIDE': ':',                # Dotted
    'Latent Diffusion': ':',      # Dotted
    'Stable Diffusion': ':',      # Dotted
    # Transformer models with dash-dot lines
    'Taming Transformers': '-.',   # Dash-dot
    'DALL-E Mini': '-.'           # Dash-dot
}

def load_spectra(model_dir):
    """Load the spectral data from a model's high quality directory."""
    base_path = Path('CompressedAnalysis') / model_dir / 'high_quality'
    data_file = base_path / 'data.npz'
    
    if not data_file.exists():
        raise FileNotFoundError(f"Spectral data not found for {model_dir}")
    
    data = np.load(data_file)
    return data['freq'], data['spectra_mean']

def plot_spectra(models_dict, title_suffix, output_filename):
    plt.figure(figsize=(12, 8))
    
    # First, get all real data to use as reference
    real_spectra = {}
    for model_name, model_dir in models_dict.items():
        if '(Real)' in model_name:
            try:
                freq, spectra = load_spectra(model_dir)
                real_spectra[model_name] = (freq, spectra)
            except Exception as e:
                print(f"Error loading real data for {model_name}: {str(e)}")
    
    # Plot spectra for each model
    for model_name, model_dir in models_dict.items():
        try:
            freq, spectra_mean = load_spectra(model_dir)
            
            # Plot with different styles for real vs generated
            plt.semilogy(freq, spectra_mean, 
                    label=model_name, 
                    color=COLORS[model_name],
                    linestyle=LINE_STYLES[model_name],
                    linewidth=2)
            
            # If this is a generated model, shade the area between COCO (reference) and model
            if '(Real)' not in model_name and 'COCO (Real)' in real_spectra:
                ref_freq, ref_spectra = real_spectra['COCO (Real)']
                plt.fill_between(freq, 
                               ref_spectra, 
                               spectra_mean,
                               color=COLORS[model_name],
                               alpha=0.1)
            
        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")
    
    plt.xlabel('Spatial Frequency (cycles/pixel)', fontsize=12)
    plt.ylabel('Power Spectrum (log scale)', fontsize=12)
    plt.title(f'Spectral Analysis: Real vs {title_suffix}', fontsize=14)
    
    # Set x-axis limits to focus on relevant frequency range
    plt.xlim(0, 0.5)
    
    # Set y-axis limits to start at 10^-5
    y_min = 1e-5
    # Get max value from all data
    all_spectra = []
    for model_name, model_dir in models_dict.items():
        try:
            _, spectra = load_spectra(model_dir)
            all_spectra.append(spectra)
        except Exception:
            continue
    
    all_spectra = np.concatenate(all_spectra)
    y_max = np.max(all_spectra) * 1.2  # 20% above max
    plt.ylim(y_min, y_max)
    
    # Add a grid for better readability
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Move legend outside the plot to avoid overlapping with the shaded areas
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot with extra space for the legend
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Plot GAN models
    plot_spectra(GAN_MODELS, 'GAN-Generated Images', 'high_quality_spectra_gan_comparison.png')
    print("GAN comparison plot has been saved as 'high_quality_spectra_gan_comparison.png'")
    
    # Plot Diffusion models
    plot_spectra(DIFFUSION_MODELS, 'Diffusion-Generated Images', 'high_quality_spectra_diffusion_comparison.png')
    print("Diffusion comparison plot has been saved as 'high_quality_spectra_diffusion_comparison.png'")
    
    # Plot Transformer models
    plot_spectra(TRANSFORMER_MODELS, 'Transformer-Generated Images', 'high_quality_spectra_transformer_comparison.png')
    print("Transformer comparison plot has been saved as 'high_quality_spectra_transformer_comparison.png'") 