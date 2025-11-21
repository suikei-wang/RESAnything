import argparse
import os
import shutil
import torch
import json
import glob
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config import load_config
from prompts import load_prompts
from sam_utils import sam_processing, combine_binary_masks
from generation import reference_generation, candidate_generation
from similarity import text_similarity, clip_similarity, selection

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def load_expressions(expressions_file):
    """Load expressions from text file where each line is 'imagename|expression'"""
    expressions = {}
    with open(expressions_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and '|' in line:
                parts = line.split('|', 1)
                if len(parts) == 2:
                    image_name, expression = parts
                    expressions[image_name.strip()] = expression.strip()
    return expressions

def process_single_image(model, processor, image_path, target, output_path, prompts, config):
    """Process a single image with the given target expression"""
    print(f"\nProcessing image: {os.path.basename(image_path)}")
    print(f"Target expression: {target}")
    
    # Create output directory for this image
    image_output_dir = os.path.join(output_path, os.path.splitext(os.path.basename(image_path))[0])
    os.makedirs(image_output_dir, exist_ok=True)
    
    sam_dir = os.path.join(image_output_dir, os.path.splitext(os.path.basename(image_path))[0] + '-SAM')
    cropped_dir = os.path.join(sam_dir, 'cropped')
    json_path = os.path.join(sam_dir, os.path.splitext(os.path.basename(image_path))[0] + '.json')

    # Step 1: Mask proposal generation (SAM)
    if not (os.path.exists(cropped_dir) and len(os.listdir(cropped_dir)) > 0):
        if image_path.endswith(('.jpg', '.png', '.jpeg')):
            try:
                print("Generating mask proposals...")
                sam_processing(image_path, image_output_dir, config['sam'])
                print("Done!")
            except Exception as e:
                print(f"Failed to generate mask proposals from SAM: {e}")
                return False
    else:
        print("Mask proposals already exist. Skipping SAM mask generation.")

    # Step 2: Reference and candidate text generation
    if os.path.exists(json_path):
        print("Candidate text JSON already exists. Skipping candidate generation.")
        with open(json_path, 'r') as f:
            candidate_texts_full = json.load(f)
        candidate_texts = {}
        for k, v in candidate_texts_full.items():
            if isinstance(v, dict) and 'candidate_text' in v:
                candidate_texts[k] = v['candidate_text']
            else:
                candidate_texts[k] = v
        reference_text = reference_generation(model, processor, image_path, target, prompts)
    else:
        reference_text = reference_generation(model, processor, image_path, target, prompts)
        print("Reference text generated.")
        print("Generating candidate texts...")
        candidate_texts = candidate_generation(model, processor, image_path, image_output_dir, prompts)
    
    print(f"Reference text: {reference_text}")

    # Step 3: Similarity analysis
    print("Generating similarity decision...")
    updated_data = text_similarity(model, processor, target, image_path, image_output_dir, reference_text, candidate_texts, prompts, batch_size=config['batch_size'])

    print("Calculating CLIP similarity...")
    final_data = clip_similarity(image_path, image_output_dir, reference_text, updated_data)

    # Step 4: Selection and output
    print("Selecting the output...")
    selected_masks = selection(final_data)
    if selected_masks:
        output_name = selected_masks[0]
        overlay_path = os.path.join(sam_dir, 'overlay', output_name)
        output_path_final = os.path.join(image_output_dir, 'output.png')
        shutil.copy(overlay_path, output_path_final)
        print(f"Output saved to: {output_path_final}")
        return True
    else:
        print("No suitable mask found.")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_folder", required=True, help="Path to the folder containing input images")
    parser.add_argument("--expressions_file", required=True, help="Path to the text file with expressions (format: imagename|expression)")
    parser.add_argument("--output_path", required=True, help="Path to the output directory") 
    args = parser.parse_args()

    config = load_config()
    prompts = load_prompts(config['paths']['prompts'])

    images_folder = args.images_folder
    expressions_file = args.expressions_file
    output_path = args.output_path

    # Load expressions
    print(f"Loading expressions from: {expressions_file}")
    expressions = load_expressions(expressions_file)
    print(f"Loaded {len(expressions)} expressions")

    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_folder, ext)))
        image_files.extend(glob.glob(os.path.join(images_folder, ext.upper())))
    
    print(f"Found {len(image_files)} images in folder")

    # Load model and processor once
    print("Loading model and processor...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config['model']['name'],
        torch_dtype=getattr(torch, config['model']['dtype']),
        attn_implementation=config['model']['attn_implementation'],
        device_map=config['model']['device_map'],
    )
    processor = AutoProcessor.from_pretrained(
        config['model']['name'], 
        min_pixels=256*28*28, 
        max_pixels=1280*28*28,
        use_fast=True
    )
    processor.tokenizer.padding_side = 'left'
    print("Model and processor loaded successfully")

    # Process each image
    successful_count = 0
    total_count = 0
    
    for image_path in image_files:
        image_name = os.path.basename(image_path)
        
        # Check if we have an expression for this image
        if image_name in expressions:
            target = expressions[image_name]
            total_count += 1
            
            try:
                success = process_single_image(model, processor, image_path, target, output_path, prompts, config)
                if success:
                    successful_count += 1
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
        else:
            print(f"No expression found for {image_name}, skipping...")

    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful_count}/{total_count} images")

if __name__ == "__main__":
    main()