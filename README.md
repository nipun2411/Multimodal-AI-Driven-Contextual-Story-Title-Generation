# Multimodal Story Title Generation Web App

This is a full-stack web application that uses the **BLIP End-to-End** model to generate a contextual title for a story, based on both the story's text and a corresponding image.

The app is built with a **Flask/Python** backend and **Vanilla HTML/CSS/JavaScript** frontend.

## Features

* Upload an image (JPG/PNG) and enter story text.
* Generate a contextual title using the BLIP model (FR1, FR2).
* Load random samples from the VIST dataset for quick testing (FR7).
* Computes and displays the BERTScore F1 alignment if a reference title is available (FR3, FR5).

## Technology Stack

* **Backend:** Flask, Transformers, PyTorch, Datasets, Evaluate (bert_score)
* **Frontend:** Vanilla HTML, CSS, JavaScript (no frameworks)
* **Model:** `Salesforce/blip-image-captioning-base`

## Setup

1.  **Clone the repository** (or copy these files into a new project folder).

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

3.  **Activate the environment:**
    * **Windows:** `.\venv\Scripts\activate`
    * **macOS/Linux:** `source venv/bin/activate`

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Execution

1.  **Run the Flask backend:**
    ```bash
    python app.py
    ```
    *Wait for the console to show that the model and dataset are loaded and the app is running (e.g., `Running on http://127.0.0.1:5000`).*

2.  **Open the web app:**
    Open your web browser and navigate to:
    [http://127.0.0.1:5000](http://127.0.0.1:5000)
