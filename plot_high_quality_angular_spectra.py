import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Define the models and their directories
GAN_MODELS = {
    # Real dataset
    'ImageNet (Real)': 'comparative_real_imagenet_valid',
    # GAN models
    'BigGAN': 'comparative_biggan_256',
    'ProGAN': 'comparative_progan_lsun',
    'StyleGAN2': 'comparative_stylegan2_ffhq_256x256',
    'StyleGAN3': 'comparative_stylegan3_r_ffhqu_256x256',
    'EG3D': 'comparative_eg3d'
}

DIFFUSION_MODELS = {
    # Real dataset
    'ImageNet (Real)': 'comparative_real_imagenet_valid',
    # Diffusion models
    'Guided Diffusion': 'comparative_guided-diffusion_class2image_ImageNet',
    'GLIDE': 'comparative_glide_text2img_valid',
    'Latent Diffusion': 'comparative_latent-diffusion_class2image_ImageNet',
    'Stable Diffusion': 'comparative_stable_diffusion_256'
}

TRANSFORMER_MODELS = {
    # Real dataset
    'ImageNet (Real)': 'comparative_real_imagenet_valid',
    # Transformer models
    'Taming Transformers': 'comparative_taming-transformers_class2image_ImageNet',
    'DALL-E Mini': 'comparative_dalle-mini_valid'
}

# Define colors and styles
COLORS = {
    # Real dataset
    'ImageNet (Real)': '#000000',  # Black
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
    # Real dataset
    'ImageNet (Real)': '-',           # Solid
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

def load_angular_spectra(model_dir):
    """Load the angular spectral data from a model's high quality directory."""
    base_path = Path('CompressedAnalysis') / model_dir / 'high_quality'
    data_file = base_path / 'data.npz'
    
    if not data_file.exists():
        raise FileNotFoundError(f"Angular spectral data not found for {model_dir}")
    
    data = np.load(data_file)
    return data['ang_freq'], data['ang_spectra_mean'], data['ang_spectra_var']

def compute_fisher_discriminant(model_mean, model_var, imagenet_mean, imagenet_var, epsilon=1e-10):
    """Compute Fisher's discriminant between model and ImageNet spectra."""
    # Add small epsilon to avoid division by zero
    denominator = np.sqrt(model_var + imagenet_var + epsilon)
    return (model_mean - imagenet_mean) / denominator

def plot_angular_spectra(models_dict, title_suffix, output_filename):
    plt.figure(figsize=(10, 10))
    
    # Create polar subplot
    ax = plt.subplot(111, projection='polar')
    
    # First, get ImageNet data as reference
    imagenet_dir = models_dict['ImageNet (Real)']
    try:
        ang_freq, imagenet_mean, imagenet_var = load_angular_spectra(imagenet_dir)
        # Make the angular spectra periodic for plotting
        ang_freq = np.concatenate((ang_freq, np.pi + ang_freq, 2*np.pi + ang_freq[:1]))
        imagenet_mean = np.concatenate((imagenet_mean, imagenet_mean, imagenet_mean[...,:1]))
        imagenet_var = np.concatenate((imagenet_var, imagenet_var, imagenet_var[...,:1]))
    except Exception as e:
        print(f"Error loading ImageNet data: {str(e)}")
        return
    
    # Plot spectra for each model
    for model_name, model_dir in models_dict.items():
        try:
            # Skip ImageNet as it's our reference
            if model_name == 'ImageNet (Real)':
                continue
                
            freq, model_mean, model_var = load_angular_spectra(model_dir)
            
            # Make the angular spectra periodic for plotting
            freq = np.concatenate((freq, np.pi + freq, 2*np.pi + freq[:1]))
            model_mean = np.concatenate((model_mean, model_mean, model_mean[...,:1]))
            model_var = np.concatenate((model_var, model_var, model_var[...,:1]))
            
            # Compute Fisher's discriminant
            fisher_disc = compute_fisher_discriminant(model_mean, model_var, imagenet_mean, imagenet_var)
            
            # Plot with different styles for real vs generated
            ax.plot(freq, fisher_disc, 
                   label=model_name, 
                   color=COLORS[model_name],
                   linestyle=LINE_STYLES[model_name],
                   linewidth=2)
            
            # Shade the area between zero and the Fisher discriminant
            ax.fill_between(freq, 
                          0,  # Zero line
                          fisher_disc,
                          color=COLORS[model_name],
                          alpha=0.1)
            
        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")
    
    # Add a zero line to show the ImageNet reference
    ax.plot(ang_freq, np.zeros_like(ang_freq), 
            color='black', 
            linestyle='-',  # Changed to solid line
            linewidth=2,    # Made thicker
            label='ImageNet (Reference)')
    
    # Customize the polar plot
    ax.set_title(f"{title_suffix}-Normalized by ImageNet", fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits to show the range of differences
    # Get the max absolute value and add some padding
    all_values = []
    for line in ax.get_lines():
        all_values.extend(line.get_ydata())
    max_abs = max(abs(min(all_values)), abs(max(all_values)))
    ax.set_ylim(-max_abs * 1.2, max_abs * 1.2)
    
    # Remove radial labels for cleaner look
    ax.set_rticks([])
    
    # Move legend outside the plot
    plt.legend(fontsize=10, bbox_to_anchor=(1.3, 1), loc='upper right')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot with extra space for the legend
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Plot GAN models
    plot_angular_spectra(GAN_MODELS, 'GAN Generated Images', 'high_quality_angular_spectra_gan_fisher.png')
    print("GAN Fisher's discriminant plot has been saved as 'high_quality_angular_spectra_gan_fisher.png'")
    
    # Plot Diffusion models
    plot_angular_spectra(DIFFUSION_MODELS, 'Diffusion Generated Images', 'high_quality_angular_spectra_diffusion_fisher.png')
    print("Diffusion Fisher's discriminant plot has been saved as 'high_quality_angular_spectra_diffusion_fisher.png'")
    
    # Plot Transformer models
    plot_angular_spectra(TRANSFORMER_MODELS, 'Transformer Generated Images', 'high_quality_angular_spectra_transformer_fisher.png')
    print("Transformer Fisher's discriminant plot has been saved as 'high_quality_angular_spectra_transformer_fisher.png'") 