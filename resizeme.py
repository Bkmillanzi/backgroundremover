from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import logging
from PIL import Image, ImageEnhance
import numpy as np
import io
import cv2
import torch
from torchvision import models, transforms
from backgroundremover.bg import remove
import time

app = Flask(__name__)
CORS(app)

DPI = 450
MM_TO_INCH = 25.4
logging.basicConfig(level=logging.INFO)

# Initialize GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pre-trained model for face detection
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
model.to(device)

# Function to remove background and resize image
def remove_bg_and_resize(image_data, resize_width_mm=22, resize_height_mm=17):
    model_choices = ["u2net", "u2net_human_seg", "u2netp"]
    logging.info(f"Removing background with model {model_choices[1]}")

    # Load image
    start_time = time.time()
    with Image.open(io.BytesIO(image_data)) as img:
        width, height = img.size
        logging.info(f"Original image size: {width}x{height}")
        base_size = max(width, height)

    # Remove background
    img = remove(image_data, model_name=model_choices[1], alpha_matting=True,
                 alpha_matting_foreground_threshold=200, alpha_matting_background_threshold=1,
                 alpha_matting_erode_structure_size=1, alpha_matting_base_size=base_size)
    img = Image.open(io.BytesIO(img))

    # Resize to desired size
    placeholder_width_px = int((resize_width_mm / MM_TO_INCH) * DPI)
    placeholder_height_px = int((resize_height_mm / MM_TO_INCH) * DPI)
    img.thumbnail((placeholder_width_px, placeholder_height_px), Image.LANCZOS)

    # Create transparent placeholder
    placeholder = Image.new("RGBA", (placeholder_width_px, placeholder_height_px), (255, 255, 255, 0))
    x_offset = (placeholder_width_px - img.width) // 2
    y_offset = (placeholder_height_px - img.height) // 2
    placeholder.paste(img, (x_offset, y_offset), img if img.mode == 'RGBA' else None)
    
    logging.info(f"Background removal and resizing took {time.time() - start_time:.2f} seconds")

    return placeholder

# Function to enhance contrast
def process_contrast(image_data, contrast_factor=1.5):
    img = Image.open(io.BytesIO(image_data))
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(contrast_factor)

# Function to enhance brightness
def process_brightness(image_data, brightness_factor=1.5):
    img = Image.open(io.BytesIO(image_data))
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(brightness_factor)

# Function to crop face using PyTorch model
def crop_face(image_data):
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Extract face locations
    pred_boxes = predictions[0]['boxes']
    pred_scores = predictions[0]['scores']
    
    # Assuming the first detected face with a high score
    if len(pred_boxes) == 0:
        raise ValueError("No face detected")
    
    # Get the bounding box of the highest scored face
    best_box_idx = torch.argmax(pred_scores)
    box = pred_boxes[best_box_idx].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)
    
    # Crop and resize the face image
    face_image = image.crop((x1, y1, x2, y2))
    resized_face = face_image.resize((600, 600), Image.LANCZOS)
    
    return resized_face

@app.route('/resize-remove-bg', methods=['POST'])
def resize_and_remove_bg():
    if 'file' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400
    
    file = request.files['file']
    image_data = file.read()

    # Get resize dimensions from request (optional)
    resize_width_mm = float(request.form.get('resizeWidth', 22))
    resize_height_mm = float(request.form.get('resizeHeight', 17))

    # Process the image: remove background and resize
    processed_image = remove_bg_and_resize(image_data, resize_width_mm, resize_height_mm)

    # Save final image to BytesIO
    img_io = io.BytesIO()
    processed_image.save(img_io, format='PNG', dpi=(DPI, DPI))
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/png')

@app.route('/enhance-contrast', methods=['POST'])
def enhance_contrast_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    image_data = file.read()
    contrast_factor = float(request.form.get('contrastFactor', 1.5))
    img = process_contrast(image_data, contrast_factor)

    img_io = io.BytesIO()
    img.save(img_io, format='JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')

@app.route('/enhance-brightness', methods=['POST'])
def enhance_brightness_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    image_data = file.read()
    brightness_factor = float(request.form.get('brightnessFactor', 1.5))
    img = process_brightness(image_data, brightness_factor)

    img_io = io.BytesIO()
    img.save(img_io, format='JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')

@app.route('/crop-face', methods=['POST'])
def crop_face_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    image_data = file.read()

    try:
        cropped_face_image = crop_face(image_data)
        img_io = io.BytesIO()
        cropped_face_image.save(img_io, format='JPEG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = 5000
    logging.info(f"Starting Flask server on port {port}")
    app.run(debug=True, host='0.0.0.0', port=port)
