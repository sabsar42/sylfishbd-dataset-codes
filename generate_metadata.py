import pandas as pd
from pathlib import Path
from PIL import Image
import os

def simple_rename_files(dataset_dir):
    """
    Simple rename:
    - Images: classname_001.jpg
    - Annotations: classname_bb_001.json  
    - Masks: classname_mask_001.png
    """
    dataset_dir = Path(dataset_dir)
    categories = ['boal', 'ilish', 'kalibaush', 'katla', 'koi', 
                  'mrigel', 'pabda', 'rui', 'telapia']
    
    print("\n" + "="*60)
    print("SIMPLE FILE RENAMING")
    print("="*60)
    
    for category in categories:
        image_dir = dataset_dir / 'images' / category
        annotation_dir = dataset_dir / 'annotations' / category
        mask_dir = dataset_dir / 'masks' / category
        
        if not image_dir.exists():
            continue
        
        print(f"\n{category}:")
        
        # Get all files
        images = sorted(list(image_dir.glob('*.jpg')))
        
        counter = 1
        for img_path in images:
            old_img_name = img_path.stem
            
            # Find corresponding files (handle any naming pattern)
            ann_files = list(annotation_dir.glob(f"{old_img_name}*.json"))
            mask_files = list(mask_dir.glob(f"{old_img_name}*.png"))
            
            if not ann_files or not mask_files:
                print(f"  ⚠️ Skipping {old_img_name} - missing annotation or mask")
                continue
            
            old_ann_path = ann_files[0]
            old_mask_path = mask_files[0]
            
            # New names
            new_img_name = f"{category}_{counter:03d}.jpg"
            new_ann_name = f"{category}_bb_{counter:03d}.json"
            new_mask_name = f"{category}_mask_{counter:03d}.png"
            
            # Rename
            img_path.rename(image_dir / new_img_name)
            old_ann_path.rename(annotation_dir / new_ann_name)
            old_mask_path.rename(mask_dir / new_mask_name)
            
            counter += 1
        
        print(f"  ✅ Renamed {counter-1} files")
    
    print("\n✅ Renaming complete!")


def generate_metadata(dataset_dir, output_csv='metadata.csv'):
    """Generate metadata.csv"""
    dataset_dir = Path(dataset_dir)
    categories = ['boal', 'ilish', 'kalibaush', 'katla', 'koi', 
                  'mrigel', 'pabda', 'rui', 'telapia']
    
    print("\n" + "="*60)
    print("GENERATING METADATA")
    print("="*60)
    
    metadata_list = []
    
    for category in categories:
        image_dir = dataset_dir / 'images' / category
        
        if not image_dir.exists():
            continue
        
        images = sorted(list(image_dir.glob('*.jpg')))
        
        for img_path in images:
            img_id = img_path.stem  # e.g., 'boal_001'
            number = img_id.split('_')[-1]  # e.g., '001'
            
            # Build paths
            ann_path = f"annotations/{category}/{category}_bb_{number}.json"
            mask_path = f"masks/{category}/{category}_mask_{number}.png"
            
            # Get dimensions
            with Image.open(img_path) as img:
                width, height = img.size
            
            # Get file size
            file_size_kb = round(img_path.stat().st_size / 1024, 2)
            
            metadata_list.append({
                'image_id': img_id,
                'class_name': category,
                'image_path': f"images/{category}/{img_path.name}",
                'annotation_path': ann_path,
                'mask_path': mask_path,
                'width': width,
                'height': height,
                'file_size_kb': file_size_kb
            })
    
    df = pd.DataFrame(metadata_list)
    df = df.sort_values(['class_name', 'image_id']).reset_index(drop=True)
    
    # Save
    output_path = dataset_dir / output_csv
    df.to_csv(output_path, index=False)
    
    # Summary
    print(f"\nTotal images: {len(df)}")
    print(f"\nPer class:")
    for cat in categories:
        count = len(df[df['class_name'] == cat])
        if count > 0:
            print(f"  {cat:12s}: {count}")
    
    print(f"\n✅ Saved to: {output_path}")
    print(f"\n📄 Sample:")
    print(df.head(5).to_string(index=False))
    
    return df


def main():
    dataset_dir = "/home/shadman/BR/f_sam_mask/local_syl_bd_fish_dataset"
    
    # Step 1: Rename
    simple_rename_files(dataset_dir)
    
    # Step 2: Generate metadata
    df = generate_metadata(dataset_dir)
    
    print("\n✅ Done!")
    return df


if __name__ == "__main__":
    df = main()