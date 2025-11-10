# Import necessary libraries
import torch
import base64
import io
from flask import Flask, request, jsonify, render_template
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from datasets import load_dataset
from evaluate import load as load_metric

# --- 1. Initialization and Global Setup ---

print("Starting application setup...")

# Initialize Flask app
app = Flask(__name__, template_folder='.')

# Load the BLIP model and processor
# This is done once on startup to save time on each request.
print("Loading BLIP model and processor...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)
print(f"Model loaded successfully on {DEVICE}.")

# Load the BERTScore metric
print("Loading BERTScore metric...")
BERTSCORE = load_metric("bertscore")
print("BERTScore loaded.")

# Load and preprocess the VIST dataset (FR4)
# We only load a small, shuffled subset of the test split for sample generation.
print("Loading VIST dataset...")
try:
    VIST_DATASET = load_dataset("vist", split='test').shuffle(seed=42).select(range(100))
    print("VIST dataset loaded and pre-selected.")
except Exception as e:
    print(f"Could not load VIST dataset. Samples will not be available. Error: {e}")
    VIST_DATASET = None

# --- 2. Helper Functions ---

def preprocess_vist_sample(sample):
    """
    Processes a raw VIST dataset sample into the format
    our project expects (first image, full story, reference title).
    """
    # Use the first image in the sequence
    image = sample['images'][0].convert("RGB")
    
    # Join all 5 captions to create the 'story_text' (as per project report)
    story_text = " ".join(sample['text'])
    
    # Use the first caption as the 'reference_title' (as per project report)
    reference_title = sample['text'][0]
    
    return image, story_text, reference_title

def image_to_base64(img):
    """Converts a PIL Image to a Base64 string for a JSON response."""
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

# --- 3. Flask API Endpoints ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/get_sample', methods=['GET'])
def get_sample():
    """
    (FR7: Sample Retrieval)
    Fetches a random sample from the preloaded VIST dataset.
    """
    if VIST_DATASET is None:
        return jsonify({"error": "VIST Dataset not available"}), 500

    try:
        # Get a random sample
        sample = VIST_DATASET.shuffle().select(range(1))[0]
        
        # Process it
        image, story_text, reference_title = preprocess_vist_sample(sample)
        
        # Convert image to Base64 to send to frontend
        image_b64 = image_to_base64(image)
        
        return jsonify({
            "story_text": story_text,
            "reference_title": reference_title,
            "image_base64": image_b64
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_title', methods=['POST'])
def generate_title_endpoint():
    """
    (FR2: Title Generation)
    The main API endpoint. Receives story text and an image,
    returns a generated title and BERTScore.
    """
    try:
        # Get data from the frontend's JSON request
        data = request.json
        story_text = data.get('story_text')
        base64_image = data.get('image_base64')
        reference_title = data.get('reference_title') # This may be null

        # --- Error Handling (FR6) ---
        if not story_text or not base64_image:
            return jsonify({"error": "Missing story_text or image_base64"}), 400
        
        # --- Image Preprocessing ---
        # Decode the Base64 image string
        # We split to remove the "data:image/jpeg;base64," prefix
        img_data = base64.b64decode(base64_image.split(',')[1])
        image = Image.open(io.BytesIO(img_data)).convert("RGB")

        # --- Model Inference (FR2) ---
        # 1. Preprocess inputs
        # The text is used as a "prompt" for the image captioning model
        inputs = PROCESSOR(images=image, text=story_text, return_tensors="pt").to(DEVICE)
        
        # 2. Generate output
        output_ids = MODEL.generate(**inputs, max_new_tokens=25) # 25 tokens max for a title
        
        # 3. Decode the generated title
        generated_title = PROCESSOR.decode(output_ids[0], skip_special_tokens=True)
        
        # --- BERTScore Evaluation (FR3) ---
        bertscore_f1 = None
        if reference_title:
            try:
                # Compute the BERTScore
                score_results = BERTSCORE.compute(
                    predictions=[generated_title],
                    references=[reference_title],
                    lang="en"
                )
                bertscore_f1 = score_results['f1'][0]
            except Exception as e:
                print(f"Error computing BERTScore: {e}")
                # If scoring fails, we still return the title
                pass 

        # --- Return Response (FR5) ---
        return jsonify({
            "generated_title": generated_title.strip(),
            "bertscore": bertscore_f1 # This will be null if no reference was provided
        })

    except Exception as e:*:
        print(f"Error in /generate_title: {e}")
        return jsonify({"error": f"Internal server error: {e}"}), 500

# --- 4. Run the Application ---
if __name__ == '__main__':
    print("Starting Flask server... Open http://127.0.0.1:5000 in your browser.")
    app.run(debug=True)