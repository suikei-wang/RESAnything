import os
import json
from generation import image_to_data_url
from sam_utils import combine_binary_masks
from qwen_vl_utils import process_vision_info
import torch
import clip
from PIL import Image
import numpy as np

def text_similarity(model, processor, target, image_path, output_path, reference_text, candidate_texts, prompts, batch_size=16):
    mask_dir = os.path.join(output_path, os.path.splitext(os.path.basename(image_path))[0] + '-SAM', 'cropped')
    text_comparison_results = {}
    mask_comparison_results = {}
    candidate_ids = list(candidate_texts.keys())
    candidate_text_list = [candidate_texts[cid] for cid in candidate_ids]

    # --- Batch text-text similarity ---
    text_text_prompts = [
        prompts["text_similarity"].format(target=target, reference_text=reference_text, candidate_text=candidate_text_list[j])
        for j in range(len(candidate_text_list))
    ]
    origin_image_url = image_to_data_url(image_path)
    text_text_messages = [
        [{
            "role": "user",
            "content": [
                {"type": "text", "text": text_text_prompts[j]},
                {"type": "image", "image_url": origin_image_url},
            ]
        }] for j in range(len(candidate_ids))
    ]
    for i in range(0, len(candidate_ids), batch_size):
        batch_messages = text_text_messages[i:i+batch_size]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_messages]
        image_inputs, video_inputs = process_vision_info([m[0] for m in batch_messages])
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda:0")
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for j, idx in enumerate(range(i, min(i+batch_size, len(candidate_ids)))):
            text_comparison = output_texts[j].lower()
            candidate_id = candidate_ids[idx]
            if 'yes' in text_comparison:
                text_comparison_results[candidate_id] = 1
            else:
                text_comparison_results[candidate_id] = 0

    # --- Batch text-mask similarity ---
    text_mask_prompts = [
        prompts["text_mask_similarity"].format(reference_text=reference_text, target=target)
        for _ in candidate_ids
    ]
    mask_urls = [image_to_data_url(os.path.join(mask_dir, cid)) for cid in candidate_ids]
    text_mask_messages = [
        [{
            "role": "user",
            "content": [
                {"type": "text", "text": text_mask_prompts[j]},
                {"type": "image", "image_url": mask_urls[j]},
            ]
        }] for j in range(len(candidate_ids))
    ]
    for i in range(0, len(candidate_ids), batch_size):
        batch_messages = text_mask_messages[i:i+batch_size]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_messages]
        image_inputs, video_inputs = process_vision_info([m[0] for m in batch_messages])
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda:0")
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for j, idx in enumerate(range(i, min(i+batch_size, len(candidate_ids)))):
            mask_comparison = output_texts[j].lower()
            candidate_id = candidate_ids[idx]
            if 'yes' in mask_comparison:
                mask_comparison_results[candidate_id] = 1
            else:
                mask_comparison_results[candidate_id] = 0

    updated_data = {}
    for candidate_id, candidate_info in candidate_texts.items():
        if isinstance(candidate_info, str):
            updated_data[candidate_id] = {
                "candidate_text": candidate_info,
                "text_comparison": text_comparison_results.get(candidate_id, 0),
                "mask_comparison": mask_comparison_results.get(candidate_id, 0)
            }
        else:
            updated_data[candidate_id] = candidate_info
            updated_data[candidate_id]["text_comparison"] = text_comparison_results.get(candidate_id, 0)
            updated_data[candidate_id]["mask_comparison"] = mask_comparison_results.get(candidate_id, 0)

    both_ones = [mask_id for mask_id, info in updated_data.items()
        if info.get('text_comparison') == 1 and info.get('mask_comparison') == 1]
    if len(both_ones) >= 1:
        both_ones = [item.replace('png', 'txt') for item in both_ones]
        masked_image_name = combine_binary_masks(both_ones, image_path, output_path)

    with open(os.path.join(output_path, os.path.splitext(os.path.basename(image_path))[0] + '-SAM', os.path.splitext(os.path.basename(image_path))[0]) + '.json', 'w') as outfile:
        json.dump(updated_data, outfile, indent=4)
    
    return updated_data

def truncate_text_for_clip(text, max_length=77):
    # Simple heuristic: CLIP tokenizer roughly uses 4 characters per token
    # So we can estimate the token count and truncate accordingly
    estimated_tokens = len(text) // 4
    
    if estimated_tokens <= max_length:
        return text
    
    # Truncate to approximately max_length tokens
    max_chars = max_length * 4
    truncated_text = text[:max_chars]
    
    # Further truncate if still too long (safety check)
    if len(truncated_text) > 300:  # Conservative limit
        truncated_text = truncated_text[:300]
    
    return truncated_text

def clip_similarity(image_path, output_path, reference_text, candidate_texts):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    mask_dir = os.path.join(output_path, os.path.splitext(os.path.basename(image_path))[0] + '-SAM', 'cropped')
    candidate_ids = list(candidate_texts.keys())
    candidate_text_list = [candidate_texts[cid]["candidate_text"] if isinstance(candidate_texts[cid], dict) else candidate_texts[cid] for cid in candidate_ids]
    
    try:
        # Batch text-text CLIP with truncation
        texts = [truncate_text_for_clip(reference_text)] + [truncate_text_for_clip(t) for t in candidate_text_list]
        text_tokens = clip.tokenize(texts).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        ref_feat = text_features[0:1]
        cand_feats = text_features[1:]
        text_text_clip_scores = (ref_feat @ cand_feats.T).squeeze(0).cpu().numpy()
        
        # Batch text-image CLIP
        mask_image_paths = [os.path.join(mask_dir, cid) for cid in candidate_ids]
        images = [preprocess(Image.open(p)).unsqueeze(0) for p in mask_image_paths]
        images = torch.cat(images, dim=0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        text_img_clip_scores = (image_features @ ref_feat.T).squeeze().cpu().numpy()
        
    except Exception as e:
        print(f"CLIP processing error: {e}")
        # Fallback to neutral scores if CLIP fails
        text_text_clip_scores = np.full(len(candidate_ids), 0.5)
        text_img_clip_scores = np.full(len(candidate_ids), 0.5)
    
    updated_data = {}
    for i, candidate_id in enumerate(candidate_ids):
        candidate_info = candidate_texts[candidate_id]
        updated_data[candidate_id] = {
            **(candidate_info if isinstance(candidate_info, dict) else {"candidate_text": candidate_info}),
            "text_text_clip": float(text_text_clip_scores[i]),
            "text_img_clip": float(text_img_clip_scores[i]),
            "avg_clip": (float(text_text_clip_scores[i]) + float(text_img_clip_scores[i])) / 2
        }
    with open(os.path.join(output_path, os.path.splitext(os.path.basename(image_path))[0] + '-SAM', os.path.splitext(os.path.basename(image_path))[0]) + '.json', 'w') as outfile:
        json.dump(updated_data, outfile, indent=4)
    return updated_data

def selection(json_data):
    both_ones = [(mask_id, info.get('avg_clip', 0))
                    for mask_id, info in json_data.items()
                    if info.get('text_comparison') == 1 and info.get('mask_comparison') == 1]
    if both_ones and len(both_ones) == 1:
        return [both_ones[0][0]]
    if both_ones and len(both_ones) > 1:
        max_clip = max(score for _, score in both_ones)
        best_masks = [mask_id for mask_id, score in both_ones if score == max_clip]
        return best_masks
    
    text_text_ones = [(mask_id, info.get('avg_clip', 0))
                      for mask_id, info in json_data.items()
                      if info.get('text_comparison') == 1]
    if text_text_ones:
        max_clip = max(score for _, score in text_text_ones)
        best_masks = [mask_id for mask_id, score in text_text_ones if score == max_clip]
        return best_masks
    
    text_mask_ones = [(mask_id, info.get('avg_clip', 0))
                      for mask_id, info in json_data.items()
                      if info.get('mask_comparison') == 1]
    if text_mask_ones:
        max_clip = max(score for _, score in text_mask_ones)
        best_masks = [mask_id for mask_id, score in text_mask_ones if score == max_clip]
        return best_masks
    
    max_clip_score = max(info.get('avg_clip', -float('inf'))
                         for info in json_data.values())
    clip_best = [mask_id for mask_id, info in json_data.items()
                 if info.get('avg_clip', -float('inf')) == max_clip_score]
    if clip_best:
        return clip_best
    return []