import os
import glob
from PIL import Image
import argparse
from tqdm import tqdm

def compress_image(img, output_path, quality, subsampling=0, optimize=False):
    """
    JPEG compression with essential parameters.
    
    Args:
        img: PIL Image object
        output_path: path to save the compressed image
        quality: JPEG quality (0-100)
        subsampling: Chroma subsampling (0=4:4:4, 2=4:2:0)
        optimize: Whether to optimize the Huffman tables
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    return img.save(
        output_path,
        format='JPEG',
        quality=quality,
        subsampling=subsampling,
        optimize=optimize
    )

def process_testset_for_compression():
    """
    Process images with 4 essential compression configurations:
    1. High quality, no subsampling (best quality)
    2. High quality, with subsampling (balanced)
    3. Medium quality, no subsampling (medium compression)
    4. Low quality, with subsampling (high compression)
    """
    testset_path = './TestSet'
    compressed_base_path = './CompressedTestSet'
    
    if not os.path.exists(testset_path):
        print(f"Error: TestSet directory not found at {testset_path}")
        return
    
    # Define the 4 essential compression configurations
    compression_configs = [
        {'name': 'high_quality', 'quality': 95, 'subsampling': 0},  # Best quality
        {'name': 'high_quality_subsampled', 'quality': 95, 'subsampling': 2},  # Balanced
        {'name': 'medium_quality', 'quality': 75, 'subsampling': 0},  # Medium compression
        {'name': 'high_compression', 'quality': 60, 'subsampling': 2}  # High compression
    ]
    
    # Process each model folder in TestSet
    for model_name in os.listdir(testset_path):
        model_path = os.path.join(testset_path, model_name)
        if not os.path.isdir(model_path):
            continue
            
        print(f"\nProcessing {model_name}...")
        
        # Create corresponding folders in CompressedTestSet for each compression config
        for config in compression_configs:
            compressed_model_path = os.path.join(compressed_base_path, f"{model_name}_{config['name']}")
            os.makedirs(compressed_model_path, exist_ok=True)
            
            # Get all images in the model folder
            image_files = glob.glob(os.path.join(model_path, "*.png"))
            image_files.extend(glob.glob(os.path.join(model_path, "*.jpg")))
            image_files.extend(glob.glob(os.path.join(model_path, "*.jpeg")))
            image_files.extend(glob.glob(os.path.join(model_path, "*.JPEG")))
            image_files.extend(glob.glob(os.path.join(model_path, "*.tif")))
            image_files.extend(glob.glob(os.path.join(model_path, "*.tiff")))
            
            for image_path in tqdm(image_files, desc=f"Compressing with {config['name']}"):
                try:
                    img = Image.open(image_path)
                    output_filename = os.path.basename(image_path)
                    # Change extension to .jpg for compressed images
                    output_filename = os.path.splitext(output_filename)[0] + '.jpg'
                    output_path = os.path.join(compressed_model_path, output_filename)
                    
                    compress_image(
                        img, 
                        output_path,
                        quality=config['quality'],
                        subsampling=config['subsampling']
                    )
                    
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    process_testset_for_compression() 