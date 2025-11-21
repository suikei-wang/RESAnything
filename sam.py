import argparse
import os
import glob
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from pathlib import Path

def sam_processing_single(image_path, output_path, sam_config):
    '''
    Generate SAM mask proposals for a single image
    '''
    sam_checkpoint = sam_config.get('checkpoint', 'sam_vit_h_4b8939.pth')
    model_type = sam_config.get('model_type', 'vit_h')
    device = sam_config.get('device', 'cuda')
    
    # Load SAM model
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Failed to load image: {image_path}")
        return False
    
    # Configure mask generator
    min_mask_percentage = sam_config.get('min_mask_percentage', 0.004)
    total_image_area = image.shape[0] * image.shape[1]
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=sam_config.get('points_per_side', 16),
        pred_iou_thresh=sam_config.get('pred_iou_thresh', 0.92),
        stability_score_thresh=sam_config.get('stability_score_thresh', 0.92),
        crop_n_layers=sam_config.get('crop_n_layers', 2),
        crop_n_points_downscale_factor=sam_config.get('crop_n_points_downscale_factor', 2),
        min_mask_region_area=total_image_area * min_mask_percentage
    )
    min_area = total_image_area * min_mask_percentage
    
    # Generate masks
    masks = mask_generator.generate(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Create output directory
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    sam_save_dir = os.path.join(output_path, image_name + '-SAM')
    os.makedirs(sam_save_dir, exist_ok=True)
    os.makedirs(os.path.join(sam_save_dir, 'cropped'), exist_ok=True)
    os.makedirs(os.path.join(sam_save_dir, 'overlay'), exist_ok=True)
    
    # Sort masks by area
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    # Process each mask
    for i, ann in enumerate(sorted_masks):
        m = ann['segmentation']
        if np.sum(m) >= min_area:
            # Save mask image
            img = np.zeros((m.shape[0], m.shape[1], 4))
            img[:, :, 3] = 1
            img[m] = [1, 1, 1, 1]
            mask_filename = os.path.join(sam_save_dir, f"{i}.png")
            plt.imsave(mask_filename, img[:, :, :3], cmap='gray')

            # Save overlay mask
            b_mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
            _, b_mask = cv2.threshold(b_mask, 127, 255, cv2.THRESH_BINARY)
            mask_3ch = cv2.merge((b_mask.astype(image.dtype), b_mask.astype(image.dtype), b_mask.astype(image.dtype)))
            overlay_color_mask = cv2.multiply(mask_3ch, (0, 0, 255), dtype=cv2.CV_64F) / 255.0
            overlay = cv2.addWeighted(image.astype(np.float32), 0.5, overlay_color_mask.astype(np.float32), 0.5, 0)
            overlay = overlay.astype(np.uint8)
            overlay_filename = os.path.join(sam_save_dir, "overlay", f"{i}.png")
            cv2.imwrite(overlay_filename, overlay)

            # Save cropped mask
            cropped = cv2.bitwise_and(image, image, mask=b_mask)
            cropped_filename = os.path.join(sam_save_dir, "cropped", f"{i}.png")
            cv2.imwrite(cropped_filename, cropped)
    
    print(f"Generated {len([m for m in sorted_masks if np.sum(m['segmentation']) >= min_area])} masks for {os.path.basename(image_path)}")
    return True

def process_folder(images_folder, output_path, sam_config):
    """
    Process all images in a folder and generate SAM masks
    """
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_folder, ext)))
        image_files.extend(glob.glob(os.path.join(images_folder, ext.upper())))
    
    print(f"Found {len(image_files)} images in folder: {images_folder}")
    
    if len(image_files) == 0:
        print("No images found in the specified folder.")
        return
    
    # Process each image
    successful_count = 0
    total_count = len(image_files)
    
    for i, image_path in enumerate(image_files, 1):
        image_name = os.path.basename(image_path)
        print(f"\nProcessing image {i}/{total_count}: {image_name}")
        
        try:
            success = sam_processing_single(image_path, output_path, sam_config)
            if success:
                successful_count += 1
                print(f"✓ Successfully processed: {image_name}")
            else:
                print(f"✗ Failed to process: {image_name}")
        except Exception as e:
            print(f"✗ Error processing {image_name}: {e}")
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful_count}/{total_count} images")

def main():
    parser = argparse.ArgumentParser(description="Generate SAM segmentation masks for all images in a folder")
    parser.add_argument("--images_folder", required=True, help="Path to the folder containing input images")
    parser.add_argument("--output_path", required=True, help="Path to the output directory")
    parser.add_argument("--sam_checkpoint", default="sam_vit_h_4b8939.pth", help="Path to SAM checkpoint file")
    parser.add_argument("--model_type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"], help="SAM model type")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--min_mask_percentage", type=float, default=0.004, help="Minimum mask area as percentage of image")
    parser.add_argument("--points_per_side", type=int, default=16, help="Number of points per side for mask generation")
    parser.add_argument("--pred_iou_thresh", type=float, default=0.92, help="Prediction IoU threshold")
    parser.add_argument("--stability_score_thresh", type=float, default=0.92, help="Stability score threshold")
    parser.add_argument("--crop_n_layers", type=int, default=2, help="Number of crop layers")
    parser.add_argument("--crop_n_points_downscale_factor", type=int, default=2, help="Crop points downscale factor")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # SAM configuration
    sam_config = {
        'checkpoint': args.sam_checkpoint,
        'model_type': args.model_type,
        'device': args.device,
        'min_mask_percentage': args.min_mask_percentage,
        'points_per_side': args.points_per_side,
        'pred_iou_thresh': args.pred_iou_thresh,
        'stability_score_thresh': args.stability_score_thresh,
        'crop_n_layers': args.crop_n_layers,
        'crop_n_points_downscale_factor': args.crop_n_points_downscale_factor
    }
    
    print("SAM Configuration:")
    for key, value in sam_config.items():
        print(f"  {key}: {value}")
    
    # Process the folder
    process_folder(args.images_folder, args.output_path, sam_config)

if __name__ == "__main__":
    main()
