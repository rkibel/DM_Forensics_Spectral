##Spectral Analysis to Determine Synthetic Image Generation**

###Instructions

1. Follow dataset download and setup here:

   https://github.com/grip-unina/DMimageDetection/tree/main

   https://github.com/grip-unina/SyntheticImagesAnalysis/tree/main
2. Obtain DenoiserWeight and TestSet folders (TestSet must contain COCO, UCID, and ImageNet images)
3. For each desired model, run compress_images.py and save to folder CompressedTestSet (for each model, this will save high_compression, high_quality, high_quality_subsampled, medium_quality different JPEG compressions of the images)
4. For each desired model in CompressedTestSet (NOT JPEG compression type, just model) run generate_compressed_images.py to obtain fft, acor, and spectral data in a new CompressedAnalysis folder
5. Plot_high_quality_(angular)_spectra will plot your code from the CompressedAnalysis folder
