from flask import Flask, request, jsonify
import os
import uuid
import logging
import time
import json
import numpy as np
from werkzeug.utils import secure_filename
from paddlex import create_model

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "pdf"}
OUTPUT_FOLDER = "./output"
MODEL_NAME = "PP-DocLayout-L"

# Get environment variables
USE_GPU = os.environ.get("USE_GPU", "false").lower() == "true"

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Initialize the model
def load_model():
    try:
        logger.info(f"Loading model: {MODEL_NAME} on {'GPU' if USE_GPU else 'CPU'}")
        model = create_model(model_name=MODEL_NAME)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Check if the file extension is allowed
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to convert result to JSON-serializable format
def convert_result_to_dict(res):
    # Based on the sample output, we can extract the needed fields
    try:
        # Load the JSON file that was saved to get a proper dict
        if hasattr(res, 'json_path') and os.path.exists(res.json_path):
            with open(res.json_path, 'r') as f:
                return json.load(f)
        
        # If no JSON file or it doesn't exist, manually convert
        result = {}
        
        # Add common fields
        if hasattr(res, 'input_path'):
            result['input_path'] = res.input_path
        if hasattr(res, 'page_index'):
            result['page_index'] = res.page_index
            
        # Handle boxes if present
        if hasattr(res, 'boxes'):
            boxes = []
            for box in res.boxes:
                box_dict = {}
                if hasattr(box, 'cls_id'):
                    box_dict['cls_id'] = int(box.cls_id)
                if hasattr(box, 'label'):
                    box_dict['label'] = box.label
                if hasattr(box, 'score'):
                    box_dict['score'] = float(box.score)
                if hasattr(box, 'coordinate'):
                    box_dict['coordinate'] = [float(c) for c in box.coordinate]
                boxes.append(box_dict)
            result['boxes'] = boxes
            
        return result
    except Exception as e:
        logger.error(f"Error converting result to dict: {str(e)}")
        return {"error": "Failed to convert result to JSON-serializable format"}

# Load model at startup
model = load_model()

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "gpu_enabled": USE_GPU}), 200

@app.route("/predict", methods=["POST"])
def predict():
    # Check if the post request has the file part
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        try:
            # Generate a unique filename
            filename = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)

            # Save the file
            file.save(file_path)
            logger.info(f"File saved at: {file_path}")

            # Generate unique output file paths
            output_id = str(uuid.uuid4())
            output_img_dir = os.path.join(OUTPUT_FOLDER, output_id)
            os.makedirs(output_img_dir, exist_ok=True)

            # Make prediction with timing
            logger.info(f"Starting prediction on {'GPU' if USE_GPU else 'CPU'}")
            start_time = time.time()
            output = model.predict(file_path, batch_size=1, layout_nms=True)
            end_time = time.time()
            inference_time = end_time - start_time
            logger.info(f"Prediction completed in {inference_time:.4f} seconds")
            

            # Save results
            results = []
            for idx, res in enumerate(output):
                # Save to image
                img_path = os.path.join(output_img_dir, f"output_{idx}.jpg")
                res.save_to_img(save_path=img_path)

                # Save to JSON
                json_path = os.path.join(output_img_dir, f"output_{idx}.json")
                res.save_to_json(save_path=json_path)
                
                # Read the JSON file we just saved
                try:
                    with open(json_path, 'r') as f:
                        result_data = json.load(f)
                        results.append({
                            "idx": idx,
                            "img_path": img_path,
                            "json_path": json_path,
                            "data": result_data
                        })
                except Exception as e:
                    logger.warning(f"Could not read JSON file: {str(e)}")
                    # Fall back to manual conversion
                    results.append({
                        "idx": idx,
                        "img_path": img_path,
                        "json_path": json_path,
                        "data": convert_result_to_dict(res)
                    })

            # Clean up uploaded file
            os.remove(file_path)

            # Use our custom JSON encoder when serializing the response
            return app.response_class(
                response=json.dumps({
                    "message": "Prediction successful",
                    "inference_time": inference_time,
                    "device": "GPU" if USE_GPU else "CPU",
                    "output_directory": output_img_dir,
                    "results": results
                }, cls=NumpyEncoder),
                status=200,
                mimetype='application/json'
            )

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify(
            {
                "error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            }
        ), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)