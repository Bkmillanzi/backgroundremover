import os
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from backgroundremover.bg import remove

def remove_bg(src_img_path, out_img_path):
    model_choices = ["u2net", "u2net_human_seg", "u2netp"]
    with open(src_img_path, "rb") as f:
        data = f.read()
    img = remove(data, model_name=model_choices[1],
                 alpha_matting=True,
                 alpha_matting_foreground_threshold=240,
                 alpha_matting_background_threshold=10,
                 alpha_matting_erode_structure_size=1,
                 alpha_matting_base_size=1000)
    with open(out_img_path, "wb") as f:
        f.write(img)

def process_image(image_file):
    """Function to process a single image using backgroundremover."""
    output_dir = "output_images"
    base_filename = os.path.splitext(os.path.basename(image_file))[0]
    output_file = os.path.join(output_dir, f"{base_filename}_no_bg.png")
    
    remove_bg(image_file, output_file)

    return f"Processed: {image_file} -> {output_file}"

if __name__ == '__main__':
    # Input and output directories
    input_dir = "input_images"
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
        max_workers = 6  # You can adjust this number based on your system capacity
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks to the pool
            futures = [executor.submit(process_image, image_file) for image_file in image_files]
            
            # Process results as they complete
            for future in as_completed(futures):
                print(future.result())
