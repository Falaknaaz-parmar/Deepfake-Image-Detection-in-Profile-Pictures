"""
Vercel Serverless Function - Deepfake Detection API
Optimized for Vercel deployment with demo mode support
"""

import os
import sys
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import base64
from io import BytesIO
import random

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__) + '/..')

app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')
CORS(app)

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
DEMO_MODE = os.environ.get('DEMO_MODE', 'true') == 'true'

app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_demo(image):
    """
    Demo prediction without actual model
    Returns simulated results for demonstration
    """
    # Simulate prediction (random for demo)
    is_fake = random.choice([True, False])
    confidence = random.uniform(0.75, 0.98)
    
    if is_fake:
        prediction = 'Fake'
        fake_prob = confidence
        real_prob = 1 - confidence
        warning_level = 'high' if confidence > 0.9 else 'medium'
    else:
        prediction = 'Real'
        real_prob = confidence
        fake_prob = 1 - confidence
        warning_level = 'safe'
    
    # Create a simple heatmap overlay (demo mode)
    try:
        # Create semi-transparent colored overlay
        overlay = Image.new('RGBA', image.size, (255, 0, 0, 80) if is_fake else (0, 255, 0, 80))
        demo_image = image.convert('RGBA')
        demo_result = Image.alpha_composite(demo_image, overlay)
        
        # Convert to base64
        buffered = BytesIO()
        demo_result.convert('RGB').save(buffered, format="PNG")
        heatmap_base64 = base64.b64encode(buffered.getvalue()).decode()
        heatmap = f"data:image/png;base64,{heatmap_base64}"
    except:
        heatmap = None
    
    return {
        'prediction': prediction,
        'confidence': float(confidence),
        'probabilities': {
            'Fake': float(fake_prob),
            'Real': float(real_prob)
        },
        'is_fake': is_fake,
        'warning_level': warning_level,
        'heatmap': heatmap,
        'demo_mode': True,
        'message': 'DEMO MODE: This is a simulated prediction. Deploy with trained model for real results.'
    }


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for prediction"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, BMP, WEBP'}), 400
        
        # Read and process image
        image = Image.open(file.stream).convert('RGB')
        
        # Get prediction (demo mode)
        if DEMO_MODE:
            result = predict_demo(image)
        else:
            # This would be the real model prediction
            return jsonify({'error': 'Model not loaded. Please enable real prediction mode.'}), 503
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'demo_mode': DEMO_MODE,
        'message': 'Demo Mode Active' if DEMO_MODE else 'Production Mode'
    }), 200


# Vercel serverless function handler
def handler(event, context):
    """Vercel serverless handler"""
    return app(event, context)


# Export for Vercel
app = app
