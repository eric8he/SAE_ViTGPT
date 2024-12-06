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
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize global variables
global_data = None
num_neurons = None
nonzero_neurons = None

# Get number of neurons from first activation tensor
def get_num_neurons():
    first_activation = global_data[0][1]
    if first_activation.dim() == 2 and first_activation.size(0) == 1:
        return first_activation.size(1)
    else:
        return first_activation.size(0)

# Pre-compute nonzero neurons
def get_nonzero_neurons():
    # First, reshape all activations into a consistent format (neurons x examples)
    all_acts = []
    for _, acts in global_data:
        if acts.dim() == 2 and acts.size(0) == 1:
            all_acts.append(acts[0])  # Shape: (num_neurons,)
        else:
            all_acts.append(acts)  # Shape: (num_neurons,)
    
    # Stack all examples into a single tensor (num_neurons x num_examples)
    stacked_acts = torch.stack(all_acts, dim=1)
    
    # Find maximum absolute activation per neuron
    max_activations = torch.max(torch.abs(stacked_acts), dim=1)[0]
    
    # Get indices where max activation is > 0
    nonzero_neurons = torch.where(max_activations > 0)[0].tolist()
    
    return nonzero_neurons

def get_next_nonzero_neuron(current_idx):
    """Find the next neuron with nonzero activations."""
    for idx in nonzero_neurons:
        if idx > current_idx:
            return idx
    return None

def get_prev_nonzero_neuron(current_idx):
    """Find the previous neuron with nonzero activations."""
    for idx in reversed(nonzero_neurons):
        if idx < current_idx:
            return idx
    return None

def is_zero_neuron(neuron_idx):
    """Check if a neuron has all zero activations."""
    return neuron_idx not in nonzero_neurons

def process_image(img):
    """Convert PIL image to base64 and get dimensions."""
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return {
        'url': f"data:image/png;base64,{base64.b64encode(img_byte_arr).decode('utf-8')}",
        'width': img.width,
        'height': img.height
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/neuron/<int:neuron_idx>')
def get_neuron_data(neuron_idx):
    try:
        if neuron_idx < 0 or neuron_idx >= num_neurons:
            return jsonify({"error": f"Invalid neuron index. Must be between 0 and {num_neurons-1}"}), 400
            
        # Check if this is a zero neuron
        if is_zero_neuron(neuron_idx):
            return jsonify({
                'images': [],
                'maxNeuronIndex': num_neurons - 1,
                'nonzeroNeurons': nonzero_neurons,
                'prevNeuron': get_prev_nonzero_neuron(neuron_idx),
                'nextNeuron': get_next_nonzero_neuron(neuron_idx),
                'isZero': True
            })

        # Get top activations for this neuron
        neuron_activations = []
        for idx, (_, acts) in enumerate(global_data):
            if acts.dim() == 2 and acts.size(0) == 1:
                val = acts[0, neuron_idx].item()
            else:
                val = acts[neuron_idx].item()
            neuron_activations.append((idx, val))
        
        top_indices = sorted(neuron_activations, 
                           key=lambda x: x[1], 
                           reverse=True)[:9]
        
        processed_images = []
        for idx, activation_value in top_indices:
            img = global_data[idx][0]  # This is already a PIL Image
            img_data = process_image(img)
            img_data['activation'] = activation_value
            processed_images.append(img_data)

        return jsonify({
            'images': processed_images,
            'maxNeuronIndex': num_neurons - 1,
            'nonzeroNeurons': nonzero_neurons,
            'prevNeuron': get_prev_nonzero_neuron(neuron_idx),
            'nextNeuron': get_next_nonzero_neuron(neuron_idx),
            'isZero': False
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
    
    global global_data
    with open(args.pickle_file, 'rb') as f:
        global_data = pickle.load(f)
        
    global num_neurons
    num_neurons = get_num_neurons()
        
    global nonzero_neurons
    nonzero_neurons = get_nonzero_neurons()

    app.run(debug=True, port=8080)

if __name__ == '__main__':
    main()
