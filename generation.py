from qwen_vl_utils import process_vision_info
from PIL import Image
from io import BytesIO
import base64
import os
import json

def image_to_data_url(image_path):
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"

def reference_generation(model, processor, image_path, target, prompts):
    prompt = prompts["reference"].format(target=target)
    # prompt = prompts["reference_standard"].format(target=target)
    image_url = image_to_data_url(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image_url": image_url}
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inpupts, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inpupts,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to("cuda:0")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    reference_text = output_text[0]
    return reference_text

def candidate_generation(model, processor, image_path, output_path, prompts, batch_size=16):
    ###########
    # You may adjust this for better results (may not the optimal prompt)
    prompt = prompts['candidate']
    ###########
    mask_dir = os.path.join(output_path, os.path.splitext(os.path.basename(image_path))[0] + '-SAM', 'cropped')
    mask_images = [f for f in os.listdir(mask_dir) if f.endswith('.jpg') or f.endswith('.png')]
    mask_images_path = [os.path.join(mask_dir, mask_image) for mask_image in mask_images]

    prompts_list = [prompt] * len(mask_images_path)
    mask_urls = [image_to_data_url(mask_image) for mask_image in mask_images_path]
    candidate_texts = {}

    for i in range(0, len(mask_images_path), batch_size):
        batch_prompts = prompts_list[i:i+batch_size]
        batch_urls = mask_urls[i:i+batch_size]
        messages_batch = [
            [{
                "role": "user",
                "content": [
                    {"type": "text", "text": batch_prompts[j]},
                    {"type": "image", "image_url": batch_urls[j]}
                ]
            }] for j in range(len(batch_prompts))
        ]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_batch]
        image_inputs, video_inputs = process_vision_info([m[0] for m in messages_batch])
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
        for j, mask_image in enumerate(mask_images_path[i:i+batch_size]):
            candidate_texts[os.path.basename(mask_image)] = output_texts[j]

    with open(os.path.join(output_path, os.path.splitext(os.path.basename(image_path))[0] + '-SAM', os.path.splitext(os.path.basename(image_path))[0]) + '.json', 'w') as outfile:
        json.dump(candidate_texts, outfile, indent=4)

    return candidate_texts