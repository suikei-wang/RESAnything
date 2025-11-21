import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def sam_processing(image_path, output_path, sam_config):
    sam_checkpoint = sam_config['checkpoint']
    model_type = sam_config['model_type']
    device = sam_config['device']

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)
    image = cv2.imread(image_path)

    min_mask_percentage = sam_config['min_mask_percentage']
    total_image_area = image.shape[0] * image.shape[1]
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=sam_config['points_per_side'],
        pred_iou_thresh=sam_config['pred_iou_thresh'],
        stability_score_thresh=sam_config['stability_score_thresh'],
        crop_n_layers=sam_config['crop_n_layers'],
        crop_n_points_downscale_factor=sam_config['crop_n_points_downscale_factor'],
        min_mask_region_area=total_image_area * min_mask_percentage
    )
    min_area = total_image_area * min_mask_percentage

    masks = mask_generator.generate(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    sam_save_dir = os.path.join(output_path, image_name + '-SAM')
    os.makedirs(sam_save_dir, exist_ok=True)
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    base_image = np.zeros((sorted_masks[0]['segmentation'].shape[0], sorted_masks[0]['segmentation'].shape[1], 4))
    base_image[:, :, 3] = 1
    for i, ann in enumerate(sorted_masks):
        m = ann['segmentation']
        if np.sum(m) >= min_area:
            img = base_image.copy()
            img[m] = [1, 1, 1, 1]
            mask_filename = os.path.join(sam_save_dir, f"{i}.png")
            plt.imsave(mask_filename, img[:, :, :3], cmap='gray')

            # save overlay mask
            b_mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
            _, b_mask = cv2.threshold(b_mask, 127, 255, cv2.THRESH_BINARY)
            mask_3ch = cv2.merge((b_mask.astype(image.dtype),) * 3)
            overlay_color_mask = cv2.multiply(mask_3ch, (0, 0, 255), dtype=cv2.CV_64F) / 255.0
            overlay = cv2.addWeighted(image.astype(np.float32), 0.5, overlay_color_mask.astype(np.float32), 0.5, 0)
            overlay = overlay.astype(np.uint8)
            os.makedirs(os.path.join(sam_save_dir, "overlay"), exist_ok=True)
            overlay_filename = os.path.join(sam_save_dir, "overlay", f"{i}.png")
            cv2.imwrite(overlay_filename, overlay)

            # save mask cropped
            cropped = cv2.bitwise_and(image, image, mask=b_mask)
            os.makedirs(os.path.join(sam_save_dir, "cropped"), exist_ok=True)
            cropped_filename = os.path.join(sam_save_dir, "cropped", f"{i}.png")
            cv2.imwrite(cropped_filename, cropped)

            # save binary mask txt
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons = [np.array(polygon).squeeze() for polygon in contours]
            converted_polygons = []
            for polygon in polygons:
                polygon_list = polygon.tolist()
                converted_polygon = [coord for pair in polygon_list for coord in pair]
                converted_polygons.append(converted_polygon)
            txt_file = os.path.join(sam_save_dir, f"{i}.txt")
            np.savetxt(txt_file, m.astype(np.uint8), fmt='%d', delimiter=',')

def combine_binary_masks(mask_files, image_path, output_path):
    mask_dir = os.path.join(output_path, os.path.splitext(os.path.basename(image_path))[0] + '-SAM')
    mask_files = [os.path.join(mask_dir, mask_file) for mask_file in mask_files]
    if len(mask_files) < 1:
        raise ValueError("No mask files provided")
    
    with open(mask_files[0], 'r') as f1:
        first_line = f1.readline().strip()
        delimiter = ',' if ',' in first_line else ' '
        f1.seek(0)
        
        mask_1 = []
        for line in f1:
            cleaned = line.strip()
            if ',' in cleaned:
                row = [int(x.strip()) for x in cleaned.split(',')]
            else:
                row = [int(x.strip()) for x in cleaned.split()]
            mask_1.append(row)
    
    height = len(mask_1)
    width = len(mask_1[0]) if height > 0 else 0
    combined_mask_array = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            combined_mask_array[i, j] = 1 if mask_1[i][j] else 0

    for mask_file in mask_files[1:]:
        current_mask = []
        with open(mask_file, 'r') as f:
            for line in f:
                cleaned = line.strip()
                if ',' in cleaned:
                    row = [int(x.strip()) for x in cleaned.split(',')]
                else:
                    row = [int(x.strip()) for x in cleaned.split()]
                current_mask.append(row)
        
        if len(current_mask) != height or len(current_mask[0]) != width:
            raise ValueError(f"Mask {mask_file} dimensions do not match the first mask!")

        for i in range(height):
            for j in range(width):
                if current_mask[i][j]:
                    combined_mask_array[i, j] = 1

    combined_mask = []
    for i in range(height):
        row = []
        for j in range(width):
            row.append(str(combined_mask_array[i, j]))
        combined_mask.append(delimiter.join(row))
    
    mask_names = [os.path.splitext(os.path.basename(mask))[0] for mask in mask_files]
    combined_mask_name = "_".join(mask_names) + ".txt"
    

    output_sam_dir = os.path.join(output_path, os.path.splitext(os.path.basename(image_path))[0] + '-SAM')
    output_cropped_dir = os.path.join(output_sam_dir, 'cropped')
    os.makedirs(output_sam_dir, exist_ok=True)
    os.makedirs(output_cropped_dir, exist_ok=True)
    

    with open(os.path.join(output_sam_dir, combined_mask_name), 'w') as f_out:
        f_out.write('\n'.join(combined_mask))
    

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
        
    if combined_mask_array.shape != image.shape[:2]:
        combined_mask_array = cv2.resize(
            combined_mask_array, 
            (image.shape[1], image.shape[0]), 
            interpolation=cv2.INTER_NEAREST
        )
        
    masked_image = np.zeros_like(image)
    masked_image = cv2.bitwise_and(image, image, mask=combined_mask_array)
    
    masked_image_name = "_".join(mask_names) + ".png"

    cv2.imwrite(os.path.join(output_cropped_dir, masked_image_name), masked_image)
    
    return masked_image_name