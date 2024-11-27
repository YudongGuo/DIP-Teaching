import os
from PIL import Image
import argparse
from tqdm import tqdm
from pathlib import Path

def resize_images(input_folder, output_folder, size):
    """
    Resize all images in input_folder and save to output_folder
    Args:
        input_folder: path to input folder
        output_folder: path to output folder
        size: tuple of (width, height) or single int for shorter side
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all files in input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    # Process each image
    for img_file in tqdm(image_files, desc="Resizing images"):
        try:
            # Open image
            img_path = os.path.join(input_folder, img_file)
            img = Image.open(img_path)
            
            # Handle different resize modes
            if isinstance(size, tuple):
                # Resize to exact size
                resized_img = img.resize(size, Image.Resampling.LANCZOS)
            else:
                # Resize maintaining aspect ratio
                w, h = img.size
                if w < h:
                    new_w = size
                    new_h = int(h * size / w)
                else:
                    new_h = size
                    new_w = int(w * size / h)
                resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Save resized image
            output_path = os.path.join(output_folder, img_file)
            resized_img.save(output_path, quality=95, optimize=True)
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")

def parse_args():
    parser = argparse.ArgumentParser(description="Resize images in a folder")
    parser.add_argument("--input", type=str, required=True,
                      help="Input folder containing images")
    parser.add_argument("--output", type=str, required=True,
                      help="Output folder for resized images")
    parser.add_argument("--width", type=int, default=None,
                      help="Target width")
    parser.add_argument("--height", type=int, default=None,
                      help="Target height")
    parser.add_argument("--short_side", type=int, default=None,
                      help="Target size for shorter side (maintains aspect ratio)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Determine resize mode
    if args.width is not None and args.height is not None:
        size = (args.width, args.height)
    elif args.short_side is not None:
        size = args.short_side
    else:
        raise ValueError("Must specify either width and height, or short_side")
    
    # Resize images
    resize_images(args.input, args.output, size)
    print("Done!")