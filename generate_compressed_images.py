import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image
import glob
import os
import argparse
from area import rescale_area
from denoiser import get_denoiser
import random
from pathlib import Path
import cv2

def imread(filename):
    return np.asarray(Image.open(filename).convert('RGB'))/256.0

def rescale_img(img, siz):
    h, w = img.shape[:2]
    m = min(w, h)
    if m != siz:
        dim = (siz*w//m, siz*h//m)

        # resize image
        if siz < m:
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)

        h, w = img.shape[:2]

    assert min(w, h) == siz
    py = (h - siz)//2
    px = (w - siz)//2
    return img[py:(py+siz), px:(px+siz)]

def fft2_area(img, siz):
    img = np.fft.fft2(img, axes=(0, 1), norm='ortho')
    img_energy = np.abs(img)**2
    img_energy = rescale_area(rescale_area(img_energy, siz, 0), siz, 1)
    img_energy = np.fft.fftshift(img_energy, axes=(0, 1))
    return img_energy

def get_fft2(x):
    x = np.float64(x)
    x = x - np.mean(x, (-3, -2, -1), keepdims=True)
    x = x/np.sqrt(np.mean(np.abs(x**2), (-3, -2, -1), keepdims=True))

    x = np.fft.fft2(x, axes=(-3, -2), norm='ortho')
    x = np.abs(x)**2

    return x

def get_spectrum(power_spec, q_step=None):
    power_spec = np.mean(power_spec, -1)
    power_spec = power_spec / power_spec.size
    H, W = power_spec.shape
    h, w = np.meshgrid(np.fft.fftfreq(H), np.fft.fftfreq(W), indexing='ij')
    r = np.sqrt(h**2 + w**2)
    if q_step is None:
        q_step = 1.0/min(H, W)

    r_quant = np.round(r/q_step)
    freq = np.sort(np.unique(r_quant))
    y = np.asarray([np.sum(power_spec[r_quant == f]) for f in freq])

    return y, q_step*freq

def get_spectrum_angular(power_spec, num=16):
    power_spec = np.mean(power_spec, -1)
    power_spec = power_spec / power_spec.size
    H, W = power_spec.shape
    h, w = np.meshgrid(np.fft.fftfreq(H), np.fft.fftfreq(W), indexing='ij')
    r = np.sqrt(h**2 + w**2)
    
    angular = np.round(num * np.arctan2(h, w) / np.pi) % num
    ang_freq = np.sort(np.unique(angular))
    
    y = np.asarray([np.sum(power_spec[(angular==f) & (r>0.1)]) for f in ang_freq])
    
    return y, ang_freq/num

def process_compression_folder(folder_path, max_images=500):
    """
    Process a single compression folder and return both fingerprint and spectral data
    """
    print(f"\nProcessing folder: {folder_path}")
    filenames = glob.glob(os.path.join(folder_path, "*.jpg"))
    if not filenames:
        print(f"No images found in {folder_path}")
        return None
    
    random.shuffle(filenames)
    filenames = filenames[:max_images]
    
    # Process for FFT/autocorrelation analysis
    fund = get_denoiser(1, True)
    siz_fft = 222
    print("Generating FFT/autocorrelation fingerprints")
    res_fft2 = [fft2_area(fund(imread(_)), siz_fft) for _ in tqdm(filenames, desc="Processing images for FFT")]
    res_fft2_mean = np.mean(res_fft2, 0)
    res_fcorr_mean = np.fft.ifftshift(np.real(np.fft.ifft2(
        np.fft.ifftshift(res_fft2_mean, axes=(0, 1)), axes=(0, 1))), axes=(0, 1))
    
    # Normalize FFT results
    energy2 = np.mean(res_fft2_mean)
    res_fcorr_mean = res_fcorr_mean * 256 / 4 / energy2
    res_fft2_mean = res_fft2_mean / 4 / energy2
    
    # Process for spectral analysis
    siz_spectra = 256
    print("Generating spectral fingerprints")
    img_fft2 = [get_fft2(rescale_img(imread(_), siz_spectra)) for _ in tqdm(filenames, desc="Processing images for spectra")]
    
    freq = get_spectrum(img_fft2[0])[1]
    ang_freq = np.pi*get_spectrum_angular(img_fft2[0])[1]
    
    spectra = [get_spectrum(_)[0] for _ in tqdm(img_fft2, desc="Computing spectra")]
    ang_spectra = [get_spectrum_angular(_)[0] for _ in tqdm(img_fft2, desc="Computing angular spectra")]
    
    spectra_mean = np.mean(spectra, 0)
    ang_spectra_mean = np.mean(ang_spectra, 0)
    spectra_var = np.var(spectra, 0)
    ang_spectra_var = np.var(ang_spectra, 0)
    
    return {
        # FFT/autocorrelation data
        'res_fft2_mean': res_fft2_mean,
        'res_fcorr_mean': res_fcorr_mean,
        # Spectral data
        'freq': freq,
        'spectra_mean': spectra_mean,
        'spectra_var': spectra_var,
        'ang_freq': ang_freq,
        'ang_spectra_mean': ang_spectra_mean,
        'ang_spectra_var': ang_spectra_var
    }

def plot_comparative_visualizations(results, output_dir, model_name):
    """
    Create comparative visualizations for both FFT/autocorrelation and spectral analysis
    """
    compression_types = list(results.keys())
    
    # Create output directory
    figures_output_dir = os.path.join(output_dir, f"comparative_{model_name}")
    os.makedirs(figures_output_dir, exist_ok=True)
    
    # Plot comparative FFT2 visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx, comp_type in enumerate(compression_types):
        fft2_data = np.mean(results[comp_type]['res_fft2_mean'], -1)
        axes[idx].imshow(fft2_data.clip(0, 1), clim=[0, 1], extent=[-0.5, 0.5, 0.5, -0.5])
        axes[idx].set_title(f'FFT2 - {comp_type}')
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
    
    plt.tight_layout()
    fig.savefig(os.path.join(figures_output_dir, 'fft2_comparative.png'),
                bbox_inches='tight', pad_inches=0.0)
    plt.close()
    
    # Plot comparative autocorrelation visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx, comp_type in enumerate(compression_types):
        acor_data = np.mean(results[comp_type]['res_fcorr_mean'], -1)
        center_x = (acor_data.shape[1]+1)//2
        center_y = (acor_data.shape[0]+1)//2
        extent = [-center_x-1, acor_data.shape[1]-center_x,
                 acor_data.shape[0]-center_y, -center_y-1]
        
        axes[idx].imshow(acor_data.clip(-0.5, 0.5), clim=[-0.5, 0.5], extent=extent)
        axes[idx].set_xlim(-32, 32)
        axes[idx].set_ylim(-32, 32)
        axes[idx].set_title(f'Autocorrelation - {comp_type}')
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
    
    plt.tight_layout()
    fig.savefig(os.path.join(figures_output_dir, 'acor_comparative.png'),
                bbox_inches='tight', pad_inches=0.0)
    plt.close()
    
    # Plot comparative radial spectra
    fig, ax = plt.subplots(figsize=(10, 6))
    for comp_type in compression_types:
        ax.plot(results[comp_type]['freq'], 
                results[comp_type]['spectra_mean'],
                label=comp_type,
                linewidth=2)
    
    ax.set_xlabel('$freq$', fontsize=12)
    ax.set_ylabel('Power Spectrum', fontsize=12)
    ax.set_xlim([0.2, 0.5])
    ax.set_ylim([0.0, 0.0012])
    ax.grid(True)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_output_dir, 'spectra_comparative.png'),
                bbox_inches='tight', pad_inches=0.0)
    plt.close()
    
    # Plot comparative angular spectra
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'}, figsize=(10, 10))
    for comp_type in compression_types:
        ang_spectra_mean = np.concatenate((
            results[comp_type]['ang_spectra_mean'],
            results[comp_type]['ang_spectra_mean'],
            results[comp_type]['ang_spectra_mean'][...,:1]
        ), -1)
        ang_freq = np.concatenate((
            results[comp_type]['ang_freq'],
            np.pi + results[comp_type]['ang_freq'],
            2*np.pi + results[comp_type]['ang_freq'][:1]
        ), 0)
        ax.plot(ang_freq, ang_spectra_mean, label=comp_type, linewidth=2)
    
    ax.set_yticks(ax.get_yticks(), list())
    ax.grid(True)
    ax.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    fig.savefig(os.path.join(figures_output_dir, 'ang_spectra_comparative.png'),
                bbox_inches='tight', pad_inches=0.0)
    plt.close()
    
    # Save individual visualizations for each compression type
    for comp_type in compression_types:
        comp_output_dir = os.path.join(figures_output_dir, comp_type)
        os.makedirs(comp_output_dir, exist_ok=True)
        
        # Save FFT2
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(np.mean(results[comp_type]['res_fft2_mean'], -1).clip(0, 1),
                  clim=[0, 1], extent=[-0.5, 0.5, 0.5, -0.5])
        plt.title(f'FFT2 - {comp_type}')
        plt.xticks([])
        plt.yticks([])
        fig.savefig(os.path.join(comp_output_dir, 'fft2_gray.png'),
                   bbox_inches='tight', pad_inches=0.0)
        plt.close()
        
        # Save autocorrelation
        fig = plt.figure(figsize=(8, 8))
        acor_data = np.mean(results[comp_type]['res_fcorr_mean'], -1)
        center_x = (acor_data.shape[1]+1)//2
        center_y = (acor_data.shape[0]+1)//2
        extent = [-center_x-1, acor_data.shape[1]-center_x,
                 acor_data.shape[0]-center_y, -center_y-1]
        
        plt.imshow(acor_data.clip(-0.5, 0.5), clim=[-0.5, 0.5], extent=extent)
        plt.xlim(-32, 32)
        plt.ylim(-32, 32)
        plt.title(f'Autocorrelation - {comp_type}')
        plt.xticks([])
        plt.yticks([])
        fig.savefig(os.path.join(comp_output_dir, 'acor_gray.png'),
                   bbox_inches='tight', pad_inches=0.0)
        plt.close()
        
        # Save radial spectrum
        fig = plt.figure(figsize=(6, 5))
        plt.plot(results[comp_type]['freq'], 
                results[comp_type]['spectra_mean'],
                linewidth=2)
        plt.xlabel('$freq$', fontsize=10)
        plt.ylabel('Power Spectrum', fontsize=10)
        plt.xlim([0.2, 0.5])
        plt.ylim([0.0, 0.0012])
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(os.path.join(comp_output_dir, 'spectra.png'),
                   bbox_inches='tight', pad_inches=0.0)
        plt.close()
        
        # Save angular spectrum
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'}, figsize=(6, 6))
        ang_spectra_mean = np.concatenate((
            results[comp_type]['ang_spectra_mean'],
            results[comp_type]['ang_spectra_mean'],
            results[comp_type]['ang_spectra_mean'][...,:1]
        ), -1)
        ang_freq = np.concatenate((
            results[comp_type]['ang_freq'],
            np.pi + results[comp_type]['ang_freq'],
            2*np.pi + results[comp_type]['ang_freq'][:1]
        ), 0)
        ax.plot(ang_freq, ang_spectra_mean, linewidth=2)
        ax.set_yticks(ax.get_yticks(), list())
        ax.grid(True)
        plt.tight_layout()
        fig.savefig(os.path.join(comp_output_dir, 'ang_spectra.png'),
                   bbox_inches='tight', pad_inches=0.0)
        plt.close()
        
        # Save all data
        np.savez(os.path.join(comp_output_dir, 'data.npz'),
                # FFT/autocorrelation data
                res_fft2_mean=results[comp_type]['res_fft2_mean'],
                res_fcorr_mean=results[comp_type]['res_fcorr_mean'],
                # Spectral data
                freq=results[comp_type]['freq'],
                spectra_mean=results[comp_type]['spectra_mean'],
                spectra_var=results[comp_type]['spectra_var'],
                ang_freq=results[comp_type]['ang_freq'],
                ang_spectra_mean=results[comp_type]['ang_spectra_mean'],
                ang_spectra_var=results[comp_type]['ang_spectra_var'])

def process_compressed_testset(compressed_testset_path, output_dir, model_name=None, model_names=None):
    """
    Process the CompressedTestSet directory, either for all models, a specific model, or a list of models
    
    Args:
        compressed_testset_path: Path to the CompressedTestSet directory
        output_dir: Directory where to save the analysis results
        model_name: Optional specific model name to process (e.g., 'biggan_256')
        model_names: Optional comma-separated list of model names to process
    """
    if not os.path.exists(compressed_testset_path):
        print(f"Error: CompressedTestSet directory not found at {compressed_testset_path}")
        return
    
    # Get model folders to process
    if model_names is not None:
        # Process specific list of models
        model_folders = set(name.strip() for name in model_names.split(','))
        # Verify all models exist
        existing_models = set()
        for folder in os.listdir(compressed_testset_path):
            if '_high_compression' in folder:
                base_name = folder.split('_high_compression')[0]
                if base_name in model_folders:
                    existing_models.add(base_name)
        
        missing_models = model_folders - existing_models
        if missing_models:
            print(f"Warning: The following models were not found: {', '.join(missing_models)}")
        
        model_folders = existing_models
        if not model_folders:
            print("Error: None of the specified models were found")
            return
    elif model_name is not None:
        # Process single model
        model_exists = False
        for folder in os.listdir(compressed_testset_path):
            if folder.startswith(f"{model_name}_"):
                model_exists = True
                break
        if not model_exists:
            print(f"Error: No compressed folders found for model {model_name}")
            return
        model_folders = {model_name}
    else:
        # Process all models
        model_folders = set()
        for folder in os.listdir(compressed_testset_path):
            if '_high_compression' in folder:
                model_name = folder.split('_high_compression')[0]
                model_folders.add(model_name)
    
    compression_types = ['high_quality', 'high_quality_subsampled', 
                        'medium_quality', 'high_compression']
    
    for model_name in model_folders:
        print(f"\nProcessing model: {model_name}")
        results = {}
        
        # Process each compression type for this model
        for comp_type in compression_types:
            folder_path = os.path.join(compressed_testset_path, f"{model_name}_{comp_type}")
            if os.path.exists(folder_path):
                result = process_compression_folder(folder_path)
                if result is not None:
                    results[comp_type] = result
        
        if results:
            plot_comparative_visualizations(results, output_dir, model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compressed_testset_path", type=str, default="./CompressedTestSet",
                        help="Path to the CompressedTestSet directory")
    parser.add_argument("--output_dir", type=str, default="./CompressedAnalysis",
                        help="Directory where to save the analysis results")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Optional: Process only a specific model (e.g., 'biggan_256')")
    parser.add_argument("--model_names", type=str, default=None,
                        help="Optional: Process a comma-separated list of models (e.g., 'biggan_256,stylegan2_256,progan_256')")
    args = parser.parse_args()
    
    if args.model_name is not None and args.model_names is not None:
        print("Error: Cannot specify both --model_name and --model_names")
        exit(1)
    
    process_compressed_testset(args.compressed_testset_path, args.output_dir, 
                             args.model_name, args.model_names) 