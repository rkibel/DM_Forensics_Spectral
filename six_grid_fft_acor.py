import numpy as np
import matplotlib.pyplot as plt
import os

# Base path and directory names
base_path = "/mldata/wjfolder/SyntheticImagesAnalysis/output/"

dirs = [
    "biggan_256",
    "guided-diffusion_class2image_ImageNet",
    "real_imagenet_val",
    "stylegan3_t_ffhqu_256x256",
    "latent-diffusion_noise2image_FFHQ",
    "real_coco_valid"
]

titles = [
    "BigGAN 256",
    "Guided Diffusion",
    "Real ImageNet",
    "StyleGAN3 256",
    "Latent Diffusion",
    "Real COCO"
]

titles = dirs

def load_and_process_data(directory):
    """Load and process data from a directory's data.npz file"""
    data_path = os.path.join(base_path, directory, "data.npz")
    data = np.load(data_path)
    
    res_fft2_mean = data['res_fft2_mean']
    res_fcorr_mean = data['res_fcorr_mean']
    
    # Reconstruct plot parameters (same as original code)
    energy2 = np.mean(res_fft2_mean)
    res_fcorr_mean = res_fcorr_mean * 256 / 4 / energy2
    res_fft2_mean = res_fft2_mean / 4 / energy2
    
    return res_fft2_mean, res_fcorr_mean

# Load all data
all_fft_data = []
all_fcorr_data = []

for directory in dirs:
    try:
        fft_data, fcorr_data = load_and_process_data(directory)
        all_fft_data.append(fft_data)
        all_fcorr_data.append(fcorr_data)
        print(f"Successfully loaded data from {directory}")
    except Exception as e:
        print(f"Error loading data from {directory}: {e}")
        # Add None placeholders to maintain indexing
        all_fft_data.append(None)
        all_fcorr_data.append(None)

# Create FFT plot (2x3 grid)
fig_fft, axes = plt.subplots(2, 3, figsize=(15, 10))
fig_fft.suptitle("Mean FFT2 Comparison", fontsize=16, fontweight='bold')

for i, (fft_data, title) in enumerate(zip(all_fft_data, titles)):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    
    if fft_data is not None:
        # Plot mean FFT2 (grayscale)
        im = ax.imshow((np.mean(fft_data, -1)).clip(0, 1),
                      clim=[0, 1], extent=[-0.5, 0.5, 0.5, -0.5])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, f"Data not available\nfor {title}", 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='red')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# Create Autocorrelation plot (2x3 grid)
fig_autocorr, axes = plt.subplots(2, 3, figsize=(15, 10))
fig_autocorr.suptitle("Autocorrelation (IFFT of FFT2) Comparison", fontsize=16, fontweight='bold')

for i, (fcorr_data, title) in enumerate(zip(all_fcorr_data, titles)):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    
    if fcorr_data is not None:
        # Calculate extent for autocorrelation plot
        center_x = (fcorr_data.shape[1] + 1) // 2
        center_y = (fcorr_data.shape[0] + 1) // 2
        extent = [-center_x-1, fcorr_data.shape[1]-center_x,
                  fcorr_data.shape[0]-center_y, -center_y-1]
        
        # Plot autocorrelation
        im = ax.imshow(np.mean(fcorr_data, -1).clip(-0.5, 0.5),
                      clim=[-0.5, 0.5], extent=extent)
        ax.set_xlim(-32, 32)
        ax.set_ylim(-32, 32)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, f"Data not available\nfor {title}", 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='red')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# Optional: Save the plots
fig_fft.savefig('fft_comparison_grid.png', dpi=300, bbox_inches='tight')
fig_autocorr.savefig('autocorr_comparison_grid.png', dpi=300, bbox_inches='tight')

print("Plotting complete!")
