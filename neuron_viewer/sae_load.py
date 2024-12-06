from sae_lens import SAE
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, default_data_collator
import torch
from PIL import Image
import numpy as np
from datasets import load_dataset
import sys
from torch.utils.data import DataLoader
from typing import List

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

batch_size = 32  # Adjust based on your GPU memory
gen_kwargs = {"max_length": 16, "num_beams": 4}

def process_batch_images(images):
    images = [Image.fromarray(np.array(img)) for img in images]
    images = [i.convert(mode="RGB") if i.mode != "RGB" else i for i in images]
    return feature_extractor(images=images, return_tensors="pt").pixel_values.to(device)

def gather_residual_activations(model, target_layer, batch_inputs):
    target_act = None
    def gather_target_act_hook(mod, inputs, outputs):
        nonlocal target_act
        target_act = inputs[0]
        return outputs
    
    handle = model.decoder.transformer.h[target_layer].register_forward_hook(gather_target_act_hook)
    with torch.no_grad():
        a = model.generate(pixel_values=batch_inputs, **gen_kwargs)
    handle.remove()
    return target_act

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.9.hook_resid_pre",
    device="cuda:0",
)

# Create dataset
imgnet = load_dataset("imagenet-1k", split="validation", streaming=True)
batches = imgnet.batch(batch_size=32)

data = []
print(f"Processing in batches of {batch_size}")

batch_idx = 0
num_batches = 100

for batch in batches:
    images = [item for item in batch["image"]]
    batch_pixel_values = process_batch_images(images)
    
    # Get activations for the batch
    target_act = gather_residual_activations(model, 9, batch_pixel_values)
    sae_acts = sae.encode(target_act.to(torch.float32))
    
    # Store results
    for idx, (example, acts) in enumerate(zip(batch["image"], sae_acts)):
        data.append((example, acts.cpu()))

    batch_idx += 1
    if batch_idx % 10 == 0:
        print(f"Batch {batch_idx} done")
    if batch_idx > num_batches:
        print("Done!")
        break
    
    if (batch_idx + 1) * batch_size % 100 == 0:
        print(f"{(batch_idx + 1) * batch_size} examples done")
        print("filesize:", sys.getsizeof(data))

# store data file as pickle
import pickle
with open("arr.pkl", "wb") as f:
    pickle.dump(data, f)