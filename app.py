from flask import Flask, request, jsonify
import os
import uuid
import logging
import time
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
MODEL_NAME = "PP-DocLayout-L"

# Get environment variables
USE_GPU = os.environ.get("USE_GPU", "false").lower() == "true"

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


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
                results.append({
                    "page_index": idx,
                    "data": res.json  # This already handles serialization
                })

            # Clean up uploaded file
            os.remove(file_path)

            return jsonify(
                {
                    "message": "Prediction successful",
                    "inference_time": inference_time,
                    "device": "GPU" if USE_GPU else "CPU",
                    "results": results,
                }
            ), 200

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
