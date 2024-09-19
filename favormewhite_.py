import os
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from backgroundremover.bg import remove
from PIL import Image
import io
import logging
import time

# Constants
MM_TO_INCH = 25.4
DPI = 450

def remove_bg(src_img_path):
    model_choices = ["u2net", "u2net_human_seg", "u2netp"]
    with open(src_img_path, "rb") as f:
        data = f.read()
    img = remove(data, model_name=model_choices[1],
                 alpha_matting=True,
                 alpha_matting_foreground_threshold=240,
                 alpha_matting_background_threshold=10,
                 alpha_matting_erode_structure_size=1,
                 alpha_matting_base_size=1000)
    return img

def remove_bg_and_resize(image_data, resize_width_mm=22, resize_height_mm=17):
    model_choices = ["u2net", "u2net_human_seg", "u2netp"]
    logging.info(f"Removing background with model {model_choices[1]}")

    # Load image
    start_time = time.time()
    img = Image.open(io.BytesIO(image_data)).convert("RGBA")
    width, height = img.size
    logging.info(f"Original image size: {width}x{height}")

    # Remove background
    img = remove(image_data, model_name=model_choices[2], alpha_matting=True,
                 alpha_matting_foreground_threshold=240, alpha_matting_background_threshold=10,
                 alpha_matting_erode_structure_size=1, alpha_matting_base_size=max(width, height))
    img = Image.open(io.BytesIO(img)).convert("RGBA")

    # Calculate the size in pixels for the placeholder
    placeholder_width_px = int((resize_width_mm / MM_TO_INCH) * DPI)
    placeholder_height_px = int((resize_height_mm / MM_TO_INCH) * DPI)

    # Create transparent placeholder
    placeholder = Image.new("RGBA", (placeholder_width_px, placeholder_height_px), (255, 255, 255, 0))
    
    # Resize image while maintaining aspect ratio
    img.thumbnail((placeholder_width_px, placeholder_height_px), Image.LANCZOS)

    # Calculate offsets to center the image
    x_offset = (placeholder_width_px - img.width) // 2
    y_offset = (placeholder_height_px - img.height) // 2

    # Paste the image onto the placeholder
    placeholder.paste(img, (x_offset, y_offset), img if img.mode == 'RGBA' else None)
    
    logging.info(f"Background removal and resizing took {time.time() - start_time:.2f} seconds")

    return placeholder

def process_image(image_file):
    """Function to process a single image using backgroundremover."""
    output_dir = "output_images"
    base_filename = os.path.splitext(os.path.basename(image_file))[0]
    output_file = os.path.join(output_dir, f"{base_filename}_with_white_bg.png")  # Save as PNG

    try:
        # Remove background and resize to specified dimensions
        with open(image_file, "rb") as f:
            image_data = f.read()
        placeholder = remove_bg_and_resize(image_data)

        # Save the result
        placeholder.convert("RGB").save(output_file, format='PNG')  # Save as PNG for transparency

        return f"Processed: {image_file} -> {output_file}"
    except Exception as e:
        return f"Error processing {image_file}: {e}"

if __name__ == '__main__':
    # Input and output directories
    input_dir = "changamoto"
    output_dir = "output_images"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of all image files in the input directory (PNG and JPG)
    image_files = glob(os.path.join(input_dir, "*.png")) + glob(os.path.join(input_dir, "*.jpg"))

    # Check if there are any images to process
    if not image_files:
        print(f"No images found in {input_dir}")
    else:
        # Limit the number of parallel processes using max_workers
        max_workers = 4  # Adjust this number based on your system capacity
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks to the pool
            futures = [executor.submit(process_image, image_file) for image_file in image_files]
            
            # Process results as they complete
            for future in as_completed(futures):
                print(future.result())
