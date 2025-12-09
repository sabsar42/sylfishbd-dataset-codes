import json
import shutil
from pathlib import Path


def separate_fish_images(dataset_path, output_dir="multiple_sylbdfish"):
    """
    Move multiple fish images to a separate folder
    Keep only single fish in original dataset
    """

    dataset_path = Path(dataset_path)
    lfish_dir = dataset_path / "LFish-Dataset"

    # Create output directory
    output_path = dataset_path.parent / output_dir
    output_lfish = output_path / "LFish-Dataset"

    results = {'single': {}, 'multiple': {}, 'moved_files': []}

    print("\n" + "=" * 60)
    print("SEPARATING SINGLE AND MULTIPLE FISH IMAGES")
    print("=" * 60)

    # Process each category
    for category_dir in lfish_dir.iterdir():
        if not category_dir.is_dir():
            continue

        category_name = category_dir.name
        nested_dir = category_dir / category_name

        if not nested_dir.exists():
            continue

        print(f"\nProcessing {category_name}...")

        single_count = 0
        multiple_count = 0
        moved_count = 0

        # Create output category directory for multiple fish
        output_category_dir = output_lfish / category_name / category_name

        for json_file in nested_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # Count rectangles
                bbox_count = sum(1 for shape in data.get('shapes', [])
                                 if shape.get('shape_type') == 'rectangle')

                if bbox_count == 1:
                    single_count += 1

                elif bbox_count > 1:
                    multiple_count += 1

                    # Create output directory if needed
                    output_category_dir.mkdir(parents=True, exist_ok=True)

                    # Get corresponding image file
                    img_name = json_file.stem
                    img_extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']

                    img_file = None
                    for ext in img_extensions:
                        potential_img = json_file.parent / f"{img_name}{ext}"
                        if potential_img.exists():
                            img_file = potential_img
                            break

                    if img_file:
                        # Move JSON and image
                        shutil.move(str(json_file),
                                    str(output_category_dir / json_file.name))
                        shutil.move(str(img_file),
                                    str(output_category_dir / img_file.name))
                        moved_count += 1

                        results['moved_files'].append({
                            'category': category_name,
                            'image': img_file.name,
                            'json': json_file.name,
                            'bbox_count': bbox_count
                        })
                    else:
                        print(f"  ⚠️ Image not found for {json_file.name}")

            except Exception as e:
                print(f"  ❌ Error processing {json_file.name}: {e}")

        results['single'][category_name] = single_count
        results['multiple'][category_name] = multiple_count

        print(f"  ✅ Single fish (kept): {single_count}")
        print(f"  📦 Multiple fish (moved): {moved_count}")

    return results, output_path


def print_summary(results, output_path):
    """Print summary of separation"""

    print("\n" + "=" * 60)
    print("SEPARATION SUMMARY")
    print("=" * 60)

    total_single = sum(results['single'].values())
    total_multiple = sum(results['multiple'].values())
    total_all = total_single + total_multiple

    print(f"\n📊 PER-CATEGORY BREAKDOWN:")
    print("-" * 60)

    for category in sorted(results['single'].keys()):
        single = results['single'][category]
        multiple = results['multiple'][category]
        total = single + multiple

        print(f"\n{category.upper()}:")
        print(f"  Remained (single): {single}")
        print(f"  Moved (multiple): {multiple}")
        print(f"  Total: {total}")

    print("\n" + "=" * 60)
    print("OVERALL TOTALS")
    print("=" * 60)
    print(f"\n✅ Single fish (remained in original): {total_single}")
    print(f"📦 Multiple fish (moved to separate folder): {total_multiple}")
    print(f"📈 Total images: {total_all}")
    print(f"\n📂 Multiple fish moved to: {output_path}")
    print(f"📂 Single fish remain in: original dataset")

    if total_all > 0:
        print(f"\n📊 Single fish percentage: {total_single/total_all*100:.1f}%")
        print(
            f"📊 Multiple fish percentage: {total_multiple/total_all*100:.1f}%")


# Main execution
if __name__ == "__main__":
    # Set your dataset path
    dataset_path = "/home/shadman/BR/f_sam_mask/Local-SYL-BD-Fish-Dataset"

    print("\n⚠️  WARNING: This will MOVE files from the original dataset!")
    print("Make sure you have a backup before proceeding.")

    # Uncomment to run
    response = input("\nDo you want to proceed? (yes/no): ")

    if response.lower() in ['yes', 'y']:
        print("\nStarting separation...")

        results, output_path = separate_fish_images(
            dataset_path, output_dir="multiple_sylbdfish")

        print_summary(results, output_path)

        print("\n✅ Done!")
        print(f"\n📁 Structure:")
        print(f"   Original dataset: Only single fish images")
        print(f"   {output_path.name}/: Multiple fish images")

    else:
        print("\n❌ Operation cancelled.")
