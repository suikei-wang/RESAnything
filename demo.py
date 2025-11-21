import argparse
import os
import shutil
import torch
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config import load_config
from prompts import load_prompts
from sam_utils import sam_processing, combine_binary_masks
from generation import reference_generation, candidate_generation
from similarity import text_similarity, clip_similarity, selection

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    parser.add_argument("--input_expression", required=True, help="Input expression of the target")
    parser.add_argument("--output_path", required=True, help="Path to the segmentation output") 
    args = parser.parse_args()

    config = load_config()
    prompts = load_prompts(config['paths']['prompts'])

    image_path = args.image_path
    target = args.input_expression
    output_path = args.output_path
    
    # Ensure output_path is a directory
    if output_path.endswith(('.png', '.jpg', '.jpeg')):
        output_path = os.path.dirname(output_path) or '.'
    os.makedirs(output_path, exist_ok=True)


    sam_dir = os.path.join(output_path, os.path.splitext(os.path.basename(image_path))[0] + '-SAM')
    cropped_dir = os.path.join(sam_dir, 'cropped')
    json_path = os.path.join(sam_dir, os.path.splitext(os.path.basename(image_path))[0] + '.json')

    if not (os.path.exists(cropped_dir) and len(os.listdir(cropped_dir)) > 0):
        if image_path.endswith('.jpg') or image_path.endswith('.png'):
            try:
                print("Generating mask proposals...")
                sam_processing(image_path, output_path, config['sam'])
                print("Done!")
            except Exception as e:
                print(f"Failed to generate mask proposals from SAM: {e}")
    else:
        print("Mask proposals already exist. Skipping SAM mask generation.")

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
        candidate_texts = candidate_generation(model, processor, image_path, output_path, prompts)
    
    print(reference_text)

    print("Generating similarity decision...")
    updated_data = text_similarity(model, processor, target, image_path, output_path, reference_text, candidate_texts, prompts, batch_size=config['batch_size'])

    print("Calculating CLIP similarity...")
    final_data = clip_similarity(image_path, output_path, reference_text, updated_data)

    print("Selecting the output...")
    output_name = selection(final_data)[0]
    overlay_path = os.path.join(output_path, os.path.splitext(os.path.basename(image_path))[0] + '-SAM', 'overlay', output_name)
    shutil.copy(overlay_path, os.path.join(output_path, 'output.png'))
    print("Done!")

if __name__ == "__main__":
    main()