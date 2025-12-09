import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from PIL import Image
import json

def load_annotation(annotation_path):
    """Load bounding box annotation from JSON file"""
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    
    # Extract bounding box coordinates
    shapes = data['shapes']
    if shapes:
        points = shapes[0]['points']
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)
        
        return x_min, y_min, x_max, y_max
    return None

def generate_segmentation_table_images(dataset_dir, output_dir='figures/segmentation_table'):
    """Generate images for segmentation results table"""
    
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define all 9 fish classes with sample numbers
    fish_classes = [
    ('rui', 1),
    ('telapia', 2),
    ('mrigel', 3),
    ('katla', 4),
    ('boal', 5),
    ('pabda', 6),
    ('ilish', 7),
    ('koi', 8),
    ('kalibaush', 9)
]

    
    print("\n" + "="*70)
    print("GENERATING SEGMENTATION TABLE IMAGES")
    print("="*70 + "\n")
    
    for fish_class, sample_num in fish_classes:
        print(f"Processing: {fish_class.capitalize()}...")
        
        # File paths
        img_path = dataset_dir / 'images' / fish_class / f'{fish_class}_{sample_num:03d}.jpg'
        annotation_path = dataset_dir / 'annotations' / fish_class / f'{fish_class}_bb_{sample_num:03d}.json'
        mask_path = dataset_dir / 'masks' / fish_class / f'{fish_class}_mask_{sample_num:03d}.png'
        
        # Check if files exist
        if not img_path.exists():
            print(f"  ⚠️  Image not found: {img_path}")
            continue
        if not annotation_path.exists():
            print(f"  ⚠️  Annotation not found: {annotation_path}")
            continue
        if not mask_path.exists():
            print(f"  ⚠️  Mask not found: {mask_path}")
            continue
        
        # Load images
        img = plt.imread(str(img_path))
        mask = plt.imread(str(mask_path))
        
        # Load bounding box
        bbox = load_annotation(annotation_path)
        
        if bbox is None:
            print(f"  ⚠️  No bounding box found in annotation")
            continue
        
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        
        # ============================================================
        # 1. BOUNDING BOX ON FISH
        # ============================================================
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img)
        
        # Draw bounding box
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=3, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        ax.axis('off')
        ax.set_title(f'{fish_class.capitalize()} - Bounding Box', 
                     fontsize=12, fontweight='bold', pad=10)
        
        plt.tight_layout()
        bbox_output = output_dir / f'{fish_class}_bbox.png'
        plt.savefig(bbox_output, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        # ============================================================
        # 2. MASK ONLY
        # ============================================================
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # Display mask
        if len(mask.shape) == 3:
            mask_display = mask[:, :, 0]
        else:
            mask_display = mask
            
        ax.imshow(mask_display, cmap='gray')
        ax.axis('off')
        ax.set_title(f'{fish_class.capitalize()} - Segmentation Mask', 
                     fontsize=12, fontweight='bold', pad=10)
        
        plt.tight_layout()
        mask_output = output_dir / f'{fish_class}_mask.png'
        plt.savefig(mask_output, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        # ============================================================
        # 3. OVERLAP MASK ON FISH
        # ============================================================
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # Create overlay
        overlay = img.copy().astype(float)
        
        # Create mask boolean
        if len(mask.shape) == 2:
            mask_bool = mask > 0.5
        else:
            mask_bool = mask[:, :, 0] > 0.5
        
        # Create green overlay for mask region
        overlay_mask = np.zeros_like(img)
        overlay_mask[mask_bool] = [0, 255, 0]  # Green color
        
        # Blend original image with overlay
        blended = (img * 0.5 + overlay_mask * 0.5).astype(np.uint8)
        
        ax.imshow(blended)
        ax.axis('off')
        ax.set_title(f'{fish_class.capitalize()} - Mask Overlay', 
                     fontsize=12, fontweight='bold', pad=10)
        
        plt.tight_layout()
        overlay_output = output_dir / f'{fish_class}_overlay.png'
        plt.savefig(overlay_output, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        print(f"  ✅ Generated: {fish_class}_bbox.png, {fish_class}_mask.png, {fish_class}_overlay.png")
    
    print("\n" + "="*70)
    print("✅ ALL SEGMENTATION TABLE IMAGES GENERATED!")
    print(f"📁 Output directory: {output_dir}")
    print("="*70 + "\n")

def generate_latex_table():
    """Generate LaTeX code for segmentation results table"""
    
    fish_classes = [
        'boal', 'chingri', 'ilish', 'koi', 'pabda', 
        'punti', 'rui', 'shing', 'telapia'
    ]
    
    latex_code = r"""\begin{table}[h]
\centering
\caption{Sample Segmentation Results for All Nine Fish Classes}
\label{tab:segmentation_results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Fish Class} & \textbf{Bounding Box} & \textbf{Segmentation Mask} & \textbf{Mask Overlay} \\ \hline
"""
    
    # Add rows for each fish class
    for fish in fish_classes:
        fish_name = fish.capitalize()
        latex_code += f"""{fish_name} & 
\\includegraphics[width=0.25\\textwidth]{{figures/segmentation_table/{fish}_bbox.png}} & 
\\includegraphics[width=0.25\\textwidth]{{figures/segmentation_table/{fish}_mask.png}} & 
\\includegraphics[width=0.25\\textwidth]{{figures/segmentation_table/{fish}_overlay.png}} \\\\ \\hline
"""
    
    latex_code += r"""\end{tabular}%
}
\end{table}

% Note: The table shows the segmentation pipeline for each fish species:
% - Column 2: Original image with red bounding box annotation
% - Column 3: Binary segmentation mask generated by SAM
% - Column 4: Green overlay showing mask aligned with original fish image
"""
    
    return latex_code

def generate_latex_table_with_placeholders():
    """Generate LaTeX code with placeholder comments for easy editing"""
    
    fish_classes = [
        'boal', 'chingri', 'ilish', 'koi', 'pabda', 
        'punti', 'rui', 'shing', 'telapia'
    ]
    
    latex_code = r"""\begin{table}[h]
\centering
\caption{Sample Segmentation Results for All Nine Fish Classes}
\label{tab:segmentation_results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Fish Class} & \textbf{Bounding Box} & \textbf{Segmentation Mask} & \textbf{Mask Overlay} \\ \hline
"""
    
    # Add rows with placeholders
    for fish in fish_classes:
        fish_name = fish.capitalize()
        latex_code += f"""
% Row for {fish_name}
{fish_name} & 
\\includegraphics[width=0.25\\textwidth]{{figures/segmentation_table/{fish}_bbox.png}} & 
\\includegraphics[width=0.25\\textwidth]{{figures/segmentation_table/{fish}_mask.png}} & 
\\includegraphics[width=0.25\\textwidth]{{figures/segmentation_table/{fish}_overlay.png}} \\\\ \\hline
"""
    
    latex_code += r"""
\end{tabular}%
}
\end{table}

% INSTRUCTIONS FOR USE:
% 1. Make sure all PNG files are generated in figures/segmentation_table/
% 2. Upload all images to your Overleaf project in the correct folder structure
% 3. Add this table to your document where needed
% 4. Adjust width=0.25\textwidth if images appear too large or small
% 5. Use \resizebox{\textwidth}{!}{...} to fit table to page width
"""
    
    return latex_code

# Main execution
if __name__ == "__main__":
    # Set your dataset directory
    dataset_dir = "/home/shadman/BR/f_sam_mask/local_syl_bd_fish_dataset"
    
    # Generate all segmentation table images
    generate_segmentation_table_images(dataset_dir)
    
    # Generate LaTeX table code
    latex_table = generate_latex_table()
    latex_table_placeholder = generate_latex_table_with_placeholders()
    
    # Save LaTeX codes
    with open('segmentation_table.tex', 'w') as f:
        f.write(latex_table)
    
    with open('segmentation_table_placeholder.tex', 'w') as f:
        f.write(latex_table_placeholder)
    
    print("\n📄 LaTeX table codes saved:")
    print("   - segmentation_table.tex (clean version)")
    print("   - segmentation_table_placeholder.tex (with instructions)")
    
    print("\n📋 Required LaTeX packages:")
    print("   \\usepackage{graphicx}")
    print("   \\usepackage{array}")
    print("   \\usepackage{booktabs}  % Optional for better tables")
    
    print("\n💡 Usage in Overleaf:")
    print("   1. Upload all PNG files from figures/segmentation_table/ to Overleaf")
    print("   2. Maintain the same folder structure: figures/segmentation_table/")
    print("   3. Include the .tex file in your main document using \\input{segmentation_table.tex}")
    print("   4. Or copy-paste the table code directly into your document")