import json
from pathlib import Path


def count_single_fish(dataset_path):
    """Count images with exactly one bounding box"""

    dataset_path = Path(dataset_path)
    results = {}

    # Find all fish categories
    lfish_dir = dataset_path / "LFish-Dataset"

    for category_dir in lfish_dir.iterdir():
        if not category_dir.is_dir():
            continue

        category_name = category_dir.name
        nested_dir = category_dir / category_name

        if not nested_dir.exists():
            continue

        # Count JSON files with single bbox
        single_count = 0
        multiple_count = 0
        total_count = 0

        for json_file in nested_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # Count rectangles
                bbox_count = sum(1 for shape in data.get('shapes', [])
                                 if shape.get('shape_type') == 'rectangle')

                total_count += 1

                if bbox_count == 1:
                    single_count += 1
                elif bbox_count > 1:
                    multiple_count += 1

            except Exception as e:
                print(f"Error reading {json_file}: {e}")

        results[category_name] = {
            'total': total_count,
            'single': single_count,
            'multiple': multiple_count
        }

    return results


# Usage
dataset_path = "/home/shadman/BR/f_sam_mask/Local-SYL-BD-Fish-Dataset"
results = count_single_fish(dataset_path)

# Print results
print("\n" + "=" * 60)
print("SINGLE FISH IMAGE COUNT")
print("=" * 60)

total_single = 0
total_multiple = 0
total_all = 0

for category, counts in sorted(results.items()):
    print(f"\n{category.upper()}:")
    print(f"  Total: {counts['total']}")
    print(
        f"  Single fish: {counts['single']} ({counts['single']/counts['total']*100:.1f}%)"
    )
    print(f"  Multiple fish: {counts['multiple']}")

    total_single += counts['single']
    total_multiple += counts['multiple']
    total_all += counts['total']

print("\n" + "=" * 60)
print("OVERALL SUMMARY")
print("=" * 60)
print(f"Total images: {total_all}")
print(f"Single fish: {total_single} ({total_single/total_all*100:.1f}%)")
print(f"Multiple fish: {total_multiple} ({total_multiple/total_all*100:.1f}%)")
