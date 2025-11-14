import os
import json
from flask import Flask, render_template, request, jsonify


# --- Flask App Initialization ---
# The template_folder must be set to 'static' because index.html is in there.
app = Flask(__name__, static_folder="static", template_folder="static")

# --- Initialization Block ---
@app.before_first_request
def initialize():
    """Load models and skip VIST dataset download on server startup."""
    # Load Models and Metrics (BLIP and BERTScore)
    load_model_and_processor()
    load_bert_scorer()
    
    # Load VIST Dataset (Now intentionally skips download)
    load_vist_dataset() 

# --- Routes ---

@app.route("/")
def index():
    # FR1: Serve the main HTML page
    return render_template("index.html")

@app.route("/get_sample", methods=["GET"])
def get_sample():
    """FR7: Attempts to fetch a random sample, but returns a disabled message."""
    try:
        # This function will raise an exception because VIST_DATASET is None
        sample = get_random_vist_sample()
        return jsonify(sample)
    except Exception as e:
        # Catch the exception and send a clear message to the browser
        print(f"Error fetching VIST sample: {e}")
        return jsonify({
            "error": "Feature Disabled",
            "message": "The VIST Dataset feature is disabled. Please use the Manual Input tab to upload an image."
        }), 503 # 503 Service Unavailable

@app.route("/generate", methods=["POST"])
def generate():
    """FR2 & FR3: Generates a title for the image and scores it."""
    data = request.get_json()
    story_text = data.get("story_text")
    image_base64 = data.get("image_base64")
    reference_title = data.get("reference_title")

    if not all([story_text, image_base64]):
        return jsonify({"error": "Missing required data (story text or image)."}), 400

    try:
        # Call the core inference function from model_utils.py
        results = generate_title_and_score(story_text, image_base64, reference_title)
        return jsonify(results)
    except Exception as e:
        print(f"Error during title generation/scoring: {e}")
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

# --- Server Start ---
if __name__ == "__main__":
    print("--- Loading AI Models. VIST dataset download is skipped. ---")
    app.run(debug=True)
    print("Starting Flask server... Open http://127.0.0.1:5000 in your browser.")