import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamPredictor
import cv2
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import shutil

class FishMaskGenerator:
    def __init__(self, 
                 sam_checkpoint="path/to/sam_vit_h_4b8939.pth",
                 source_dir="/home/shadman/BR/f_sam_mask/Local-SYL-BD-Fish-Dataset/LFish-Dataset",
                 output_dir="/home/shadman/BR/f_sam_mask/local_syl_fish_dataset"):
        """
        Fish Mask Generator for all classes using BBox prompts
        Generates binary masks: white (255) for fish, black (0) for background
        """
        print("Loading SAM model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        
        # Initialize SAM model
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        
        print(f"SAM loaded on {self.device}")
        
        # Fish categories
        self.categories = ['boal', 'ilish', 'kalibaush', 'katla', 'koi', 
                          'mrigel', 'pabda', 'rui', 'telapia']
        
        # Quality control thresholds
        self.max_mask_to_bbox_ratio = 3.0
        self.max_mask_to_image_ratio = 0.6
        self.min_overlap_ratio = 0.5
        self.max_edge_touching_ratio = 0.3
    
    def setup_output_structure(self):
        """Create output directory structure"""
        # Create main directories
        (self.output_dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'annotations').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'masks').mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each category
        for category in self.categories:
            (self.output_dir / 'images' / category).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'annotations' / category).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'masks' / category).mkdir(parents=True, exist_ok=True)
        
        # Create classes.txt
        with open(self.output_dir / 'classes.txt', 'w') as f:
            for category in self.categories:
                f.write(f"{category}\n")
        
        print(f"✅ Directory structure created at: {self.output_dir}")
    
    def load_json_annotation(self, json_path):
        """Load bounding box from JSON"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for shape in data['shapes']:
            if shape['shape_type'] == 'rectangle':
                points = shape['points']
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                
                return {
                    'label': shape['label'],
                    'bbox': [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                }
        return None
    
    def check_edge_touching(self, mask, margin=5):
        """Check if mask touches image edges"""
        h, w = mask.shape
        
        top_edge = np.sum(mask[:margin, :])
        bottom_edge = np.sum(mask[-margin:, :])
        left_edge = np.sum(mask[:, :margin])
        right_edge = np.sum(mask[:, -margin:])
        
        total_edge = top_edge + bottom_edge + left_edge + right_edge
        total_mask = np.sum(mask)
        
        if total_mask == 0:
            return 0
        
        return total_edge / total_mask
    
    def analyze_mask_continuity(self, mask):
        """Check if mask is continuous"""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        
        if num_labels <= 1:
            return 0, 0
        
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) == 0:
            return 0, 0
        
        largest_area = np.max(areas)
        total_area = np.sum(areas)
        continuity_ratio = largest_area / total_area if total_area > 0 else 0
        
        return continuity_ratio, num_labels - 1
    
    def select_best_mask(self, masks, scores, bbox, img_shape):
        """Select best mask with quality control"""
        x_min, y_min, x_max, y_max = bbox
        bbox_area = (x_max - x_min) * (y_max - y_min)
        img_area = img_shape[0] * img_shape[1]
        
        best_idx = -1
        best_score = -1
        best_metrics = None
        
        for i, (mask, score) in enumerate(zip(masks, scores)):
            mask_area = np.sum(mask)
            
            if mask_area == 0:
                continue
            
            # Quality metrics
            metrics = {}
            
            # Size checks
            mask_to_bbox_ratio = mask_area / bbox_area
            mask_to_img_ratio = mask_area / img_area
            metrics['mask_to_bbox_ratio'] = mask_to_bbox_ratio
            metrics['mask_to_img_ratio'] = mask_to_img_ratio
            
            # Overlap with bbox
            bbox_mask = np.zeros_like(mask)
            bbox_mask[int(y_min):int(y_max), int(x_min):int(x_max)] = True
            overlap = np.sum(mask & bbox_mask) / mask_area
            metrics['overlap_ratio'] = overlap
            
            # Compactness
            if np.sum(mask) > 0:
                y_indices, x_indices = np.where(mask)
                mask_bbox_area = (x_indices.max() - x_indices.min() + 1) * (y_indices.max() - y_indices.min() + 1)
                compactness = mask_area / mask_bbox_area
                metrics['compactness'] = compactness
            else:
                metrics['compactness'] = 0
            
            # Edge touching
            edge_ratio = self.check_edge_touching(mask)
            metrics['edge_touching_ratio'] = edge_ratio
            
            # Continuity
            continuity_ratio, n_components = self.analyze_mask_continuity(mask)
            metrics['continuity_ratio'] = continuity_ratio
            metrics['n_components'] = n_components
            
            # Quality checks
            metrics['pass_bbox_ratio'] = mask_to_bbox_ratio <= self.max_mask_to_bbox_ratio
            metrics['pass_img_ratio'] = mask_to_img_ratio <= self.max_mask_to_image_ratio
            metrics['pass_overlap'] = overlap >= self.min_overlap_ratio
            metrics['pass_compactness'] = compactness > 0.3
            metrics['pass_edge_check'] = edge_ratio < self.max_edge_touching_ratio
            metrics['pass_continuity'] = continuity_ratio > 0.75
            
            # Calculate combined score with penalties
            quality_penalties = 1.0
            
            if not metrics['pass_bbox_ratio']:
                quality_penalties *= 0.1
            if not metrics['pass_img_ratio']:
                quality_penalties *= 0.1
            if not metrics['pass_overlap']:
                quality_penalties *= 0.4
            if not metrics['pass_compactness']:
                quality_penalties *= 0.6
            if not metrics['pass_edge_check']:
                quality_penalties *= 0.2
            if not metrics['pass_continuity']:
                quality_penalties *= 0.5
            
            combined_score = score * quality_penalties
            metrics['combined_score'] = combined_score
            metrics['sam_score'] = score
            
            if combined_score > best_score:
                best_score = combined_score
                best_idx = i
                best_metrics = metrics
        
        if best_idx == -1:
            return None, None, {'error': 'no_valid_mask'}
        
        return masks[best_idx], scores[best_idx], best_metrics
    
    def segment_with_bbox(self, image_rgb, bbox):
        """Generate mask using bounding box prompt"""
        self.predictor.set_image(image_rgb)
        
        bbox_array = np.array(bbox)
        
        # Get masks from SAM
        masks, scores, logits = self.predictor.predict(
            box=bbox_array,
            multimask_output=True
        )
        
        # Select best mask
        best_mask, best_score, quality_metrics = self.select_best_mask(
            masks, scores, bbox, image_rgb.shape
        )
        
        return best_mask, best_score, quality_metrics
    
    def process_single_image(self, image_path, json_path, output_paths):
        """Process a single image and save outputs"""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return False, "Failed to load image"
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotation
        annotation = self.load_json_annotation(json_path)
        if annotation is None:
            return False, "No annotation found"
        
        bbox = annotation['bbox']
        
        # Generate mask
        mask, score, quality_metrics = self.segment_with_bbox(image_rgb, bbox)
        
        if mask is None or 'error' in quality_metrics:
            return False, quality_metrics.get('error', 'Unknown error')
        
        # Check quality
        critical_checks = ['pass_bbox_ratio', 'pass_img_ratio', 'pass_overlap', 
                          'pass_edge_check', 'pass_continuity']
        failed_checks = [check for check in critical_checks if not quality_metrics.get(check, False)]
        
        if len(failed_checks) > 0:
            return False, f"Failed QC: {', '.join(failed_checks)}"
        
        # Save binary mask (white=255 for fish, black=0 for background)
        binary_mask = (mask * 255).astype(np.uint8)
        cv2.imwrite(str(output_paths['mask']), binary_mask)
        
        # Copy image
        shutil.copy2(image_path, output_paths['image'])
        
        # Copy annotation
        shutil.copy2(json_path, output_paths['annotation'])
        
        return True, quality_metrics
    
    def get_image_files(self, category):
        """Get all image and JSON files for a category"""
        category_path = self.source_dir / category / category
        
        if not category_path.exists():
            return []
        
        image_files = []
        for img_path in sorted(category_path.glob('*.jpg')):
            json_path = img_path.with_suffix('.json')
            if json_path.exists():
                image_files.append((img_path, json_path))
        
        return image_files
    
    def process_category(self, category):
        """Process all images in a category"""
        print(f"\n{'='*60}")
        print(f"Processing: {category.upper()}")
        print(f"{'='*60}")
        
        image_files = self.get_image_files(category)
        
        if len(image_files) == 0:
            print(f"⚠️  No images found for {category}")
            return
        
        print(f"Found {len(image_files)} images")
        
        success_count = 0
        failed_count = 0
        
        for idx, (img_path, json_path) in enumerate(tqdm(image_files, desc=f"{category}")):
            # Generate output filenames
            img_name = img_path.stem
            
            output_paths = {
                'image': self.output_dir / 'images' / category / f"{img_name}.jpg",
                'annotation': self.output_dir / 'annotations' / category / f"{img_name}.json",
                'mask': self.output_dir / 'masks' / category / f"{img_name}_mask.png"
            }
            
            success, result = self.process_single_image(img_path, json_path, output_paths)
            
            if success:
                success_count += 1
            else:
                failed_count += 1
        
        print(f"✅ Success: {success_count}")
        print(f"❌ Failed: {failed_count}")
        
        return success_count, failed_count
    
    def process_all_categories(self):
        """Process all fish categories"""
        self.setup_output_structure()
        
        total_success = 0
        total_failed = 0
        
        results = {}
        
        for category in self.categories:
            success, failed = self.process_category(category)
            results[category] = {'success': success, 'failed': failed}
            total_success += success
            total_failed += failed
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"\nCategory-wise results:")
        for category, res in results.items():
            print(f"  {category:12s}: ✅ {res['success']:3d}  ❌ {res['failed']:3d}")
        
        print(f"\nTotal Success: {total_success}")
        print(f"Total Failed:  {total_failed}")
        print(f"\n✅ Dataset saved to: {self.output_dir}")
        
        return results
    
    def preview_samples(self, category, n_samples=5):
        """Preview generated masks for a category"""
        mask_dir = self.output_dir / 'masks' / category
        image_dir = self.output_dir / 'images' / category
        
        mask_files = sorted(list(mask_dir.glob('*_mask.png')))[:n_samples]
        
        if len(mask_files) == 0:
            print(f"No masks found for {category}")
            return
        
        fig, axes = plt.subplots(len(mask_files), 3, figsize=(15, 5*len(mask_files)))
        
        if len(mask_files) == 1:
            axes = axes.reshape(1, -1)
        
        for i, mask_path in enumerate(mask_files):
            # Load mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            # Load corresponding image
            img_name = mask_path.stem.replace('_mask', '')
            img_path = image_dir / f"{img_name}.jpg"
            image = cv2.imread(str(img_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Original image
            axes[i, 0].imshow(image_rgb)
            axes[i, 0].set_title(f'Original\n{img_name}')
            axes[i, 0].axis('off')
            
            # Binary mask
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title('Binary Mask\n(White=Fish, Black=BG)')
            axes[i, 1].axis('off')
            
            # Overlay
            overlay = image_rgb.copy()
            color_mask = np.zeros_like(image_rgb)
            color_mask[mask > 0] = [0, 255, 0]
            overlay = cv2.addWeighted(image_rgb, 0.6, color_mask, 0.4, 0)
            
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title('Overlay')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'preview_{category}.png', dpi=150, bbox_inches='tight')
        print(f"Preview saved to: {self.output_dir}/preview_{category}.png")
        plt.show()


def main():
    """Main execution"""
    # Initialize generator
    generator = FishMaskGenerator(
        sam_checkpoint="/home/shadman/BR/f_sam_mask/sam_project/sam_vit_h_4b8939.pth",  # Update this path
        source_dir="/home/shadman/BR/f_sam_mask/Local-SYL-BD-Fish-Dataset/LFish-Dataset",
        output_dir="/home/shadman/BR/f_sam_mask/local_syl_bd_fish_dataset"
    )
    
    # Process all categories
    results = generator.process_all_categories()
    
    # Preview some results (optional)
    print("\nGenerating previews...")
    for category in ['boal', 'rui', 'koi']:
        if results.get(category, {}).get('success', 0) > 0:
            generator.preview_samples(category, n_samples=3)
    
    print("\n✅ All done!")
    
    return generator, results


if __name__ == "__main__":
    generator, results = main()