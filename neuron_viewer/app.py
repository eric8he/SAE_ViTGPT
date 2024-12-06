import os
import pickle
import numpy as np
from flask import Flask, render_template, jsonify
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
import argparse
from typing import List, Tuple, Any
import traceback
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variable to store our data
global_data: List[Tuple[Any, torch.Tensor]] = []

def load_pickle_file(filepath: str) -> None:
    """Load the pickle file containing (image, activation) pairs."""
    global global_data
    with open(filepath, 'rb') as f:
        global_data = pickle.load(f)
    logger.info(f"Loaded {len(global_data)} image-activation pairs")
    # Log the structure of the first item
    if global_data:
        logger.info(f"First item types - Image: {type(global_data[0][0])}, Activation: {type(global_data[0][1])}")
        logger.info(f"Activation shape: {global_data[0][1].shape}")
        logger.info(f"Activation device: {global_data[0][1].device}")
        logger.info(f"First activation values: {global_data[0][1][:5]}")

def get_top_images_for_neuron(neuron_idx: int, num_images: int = 9) -> List[dict]:
    """Get the top N images that most activate a given neuron."""
    global global_data
    
    try:
        logger.debug(f"Getting top {num_images} images for neuron {neuron_idx}")
        
        # Validate neuron index
        if not global_data:
            raise ValueError("No data loaded")
            
        first_activation = global_data[0][1]
        if first_activation.dim() == 2 and first_activation.size(0) == 1:
            # If shape is [1, N], take the second dimension
            num_neurons = first_activation.size(1)
        else:
            num_neurons = first_activation.size(0)
            
        if neuron_idx >= num_neurons:
            raise ValueError(f"Invalid neuron index {neuron_idx}. Must be between 0 and {num_neurons-1}")
        
        # Extract activations for the specified neuron
        neuron_activations = []
        for idx, (_, acts) in enumerate(global_data):
            if acts.dim() == 2 and acts.size(0) == 1:
                # If shape is [1, N], get the value at [0, neuron_idx]
                val = acts[0, neuron_idx].item()
            else:
                val = acts[neuron_idx].item()
            neuron_activations.append((idx, val))
        
        # Sort by activation value (descending) and get top N
        top_indices = sorted(neuron_activations, 
                            key=lambda x: x[1], 
                            reverse=True)[:num_images]
        
        results = []
        for idx, activation_value in top_indices:
            img = global_data[idx][0]  # This is already a PIL Image
            logger.debug(f"Image size: {img.size}")
            
            # Convert PIL image to base64
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')  # Save directly, no need for fromarray
            img_byte_arr = img_byte_arr.getvalue()
            img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
            
            results.append({
                'url': f'data:image/png;base64,{img_base64}',
                'activation': activation_value,
                'width': img.size[0],
                'height': img.size[1]
            })
        
        return results
    except Exception as e:
        logger.error(f"Error in get_top_images_for_neuron: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/neuron/<int:neuron_idx>')
def get_neuron_data(neuron_idx):
    try:
        logger.debug(f"Received request for neuron {neuron_idx}")
        if not global_data:
            return jsonify({"error": "No data loaded"}), 500
            
        first_activation = global_data[0][1]
        if first_activation.dim() == 2 and first_activation.size(0) == 1:
            num_neurons = first_activation.size(1)
        else:
            num_neurons = first_activation.size(0)
            
        if neuron_idx >= num_neurons:
            return jsonify({"error": f"Invalid neuron index. Must be between 0 and {num_neurons-1}"}), 400
            
        top_images = get_top_images_for_neuron(neuron_idx)
        return jsonify({
            "images": top_images,
            "maxNeuronIndex": num_neurons - 1
        })
    except Exception as e:
        logger.error(f"Error in get_neuron_data: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

def main():
    parser = argparse.ArgumentParser(description='Neuron Viewer Web App')
    parser.add_argument('pickle_file', type=str, help='Path to the pickle file containing image activations')
    args = parser.parse_args()
    
    if not os.path.exists(args.pickle_file):
        logger.error(f"Error: Pickle file '{args.pickle_file}' not found")
        exit(1)
    
    load_pickle_file(args.pickle_file)
    app.run(debug=True, port=8080)

if __name__ == '__main__':
    main()
