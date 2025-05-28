import numpy as np
import matplotlib.pyplot as plt
import os

# Define base directory and 6 target subdirectories
base_path = "/mldata/wjfolder/SyntheticImagesAnalysis/output"
dirs = [
    "new_biggan_256",
    "new_stylegan3_t_ffhqu_256x256",
    "new_guided-diffusion_class2image_ImageNet",
    "new_latent-diffusion_noise2image_FFHQ",
    "new_real_imagenet_val",
    "new_real_coco_valid"
]
titles = [
    "BigGAN 256",
    "StyleGAN3 256",
    "Guided Diffusion",
    "Latent Diffusion",
    "Real ImageNet",
    "Real COCO"
]
titles = dirs
num_classes = len(dirs)

# Initialize lists to store angular spectra data
ang_spectra_means = []
ang_spectra_vars = []

# Plot radial power spectrum
plt.figure(figsize=(7, 5))
for dirname, title in zip(dirs, titles):
    path = os.path.join(base_path, dirname, "spectra.npz")
    data = np.load(path)
    freq = data['freq']
    spectra_mean = data['spectra_mean']
    
    # Store angular spectra data for later use
    ang_spectra_means.append(data['ang_spectra_mean'])
    ang_spectra_vars.append(data['ang_spectra_var'])
    
    plt.plot(freq, spectra_mean, label=title, linewidth=2)

plt.xlabel("Frequency", fontsize=12)
plt.ylabel("Power", fontsize=12)
plt.title("Radial Power Spectrum Comparison", fontsize=14)
plt.xlim([0.2, 0.5])
plt.ylim([0.0, 0.0012])
plt.grid(True)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig('radial_power_spectrum.png', dpi=300, bbox_inches='tight')
plt.show()

# Convert lists to numpy arrays for easier manipulation
ang_spectra_means = np.array(ang_spectra_means)
ang_spectra_vars = np.array(ang_spectra_vars)

# Plot angular spectrum in polar coordinates
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(7, 6))
for i, (dirname, title) in enumerate(zip(dirs, titles)):
    path = os.path.join(base_path, dirname, "spectra.npz")
    data = np.load(path)
    ang_freq = data['ang_freq']
    ang_spectra_mean = data['ang_spectra_mean']
    
    # Extend to full 0–2π
    ang_freq_extended = np.concatenate([
        ang_freq,
        np.pi + ang_freq,
        [2 * np.pi + ang_freq[0]]
    ])
    ang_spectra_extended = np.concatenate([
        ang_spectra_mean,
        ang_spectra_mean,
        ang_spectra_mean[..., :1]
    ])
    ax.plot(ang_freq_extended, ang_spectra_extended, label=title, linewidth=2)

ax.set_yticks([])
ax.grid(True)
ax.set_title("Angular Power Spectrum Comparison", va='bottom', fontsize=14)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
plt.tight_layout()
plt.savefig('angular_power_spectrum_polar.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate Fisher Discriminant Ratio
fdr_per_class = []

for k in range(num_classes):
    # Class k mean and var
    mu_k = ang_spectra_means[k]
    var_k = ang_spectra_vars[k]
    
    # Rest mean and var (exclude class k)
    mu_rest = np.mean(np.delete(ang_spectra_means, k, axis=0), axis=0)
    var_rest = np.mean(np.delete(ang_spectra_vars, k, axis=0), axis=0)
    
    fdr_k = ((mu_k - mu_rest)**2) / (var_k + var_rest + 1e-10)
    fdr_per_class.append(fdr_k)

fdr_per_class = np.array(fdr_per_class)  # shape (6, bins)

# Load angular frequency for extending FDR arrays
# (Using the last loaded data - assumes all datasets have same ang_freq)
path = os.path.join(base_path, dirs[0], "spectra.npz")
data = np.load(path)
ang_freq = data['ang_freq']

# Create extended angular frequency array
ang_freq_extended = np.concatenate([
    ang_freq,
    np.pi + ang_freq,
    [2 * np.pi + ang_freq[0]]
])

# Extend frequency and FDR arrays for full angular range
fdr_per_class_extended = np.concatenate([
    fdr_per_class,
    fdr_per_class,
    fdr_per_class[:, :1]
], axis=1)

# Plot all 6 FDR curves in polar plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 6))
colors = plt.get_cmap('tab10')

for i in range(num_classes):
    ax.plot(ang_freq_extended, fdr_per_class_extended[i], label=dirs[i], color=colors(i))

ax.set_yticks([])
ax.grid(True)
ax.set_title("Fisher Discriminant Ratio (One-vs-Rest) of Angular Spectra", va='bottom')
ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig('fisher_discriminant_ratio_polar.png', dpi=300, bbox_inches='tight')
plt.show()
