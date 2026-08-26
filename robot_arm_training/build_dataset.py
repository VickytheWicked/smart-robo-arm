from pathlib import Path
import shutil
import random

# ============================================================
# PATHS
# ============================================================

ORIGINAL = Path.home() / "Downloads" / "Stationary Item Detector.v1i.yolo26"
TEACH = Path.home() / "robot_teach_dataset"
OUTPUT = Path.home() / "robot_arm_training"

# ============================================================
# CLASSES
# ============================================================

CLASSES = [
    "eraser",
    "glue",
    "marker",
    "pen",
    "pencil",
    "ruler",
    "scissor",
    "sharpener",
    "stapler",
    "tape"
]

# ============================================================
# CLEAN OUTPUT DATASET
# ============================================================

for split in ["train", "valid", "test"]:

    (OUTPUT / split / "images").mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT / split / "labels").mkdir(
        parents=True,
        exist_ok=True
    )

# ============================================================
# COPY ORIGINAL DATASET
# ============================================================

print("Copying original dataset...")

for split in ["train", "valid", "test"]:

    src_images = ORIGINAL / split / "images"
    src_labels = ORIGINAL / split / "labels"

    dst_images = OUTPUT / split / "images"
    dst_labels = OUTPUT / split / "labels"

    for image in src_images.iterdir():

        if not image.is_file():
            continue

        label = src_labels / f"{image.stem}.txt"

        if not label.exists():

            print(
                f"WARNING: Missing label for {image.name}"
            )

            continue

        # Prefix prevents filename collisions
        new_name = f"original_{image.name}"

        shutil.copy2(
            image,
            dst_images / new_name
        )

        shutil.copy2(
            label,
            dst_labels / f"original_{image.stem}.txt"
        )

# ============================================================
# COLLECT TEACH EXAMPLES
# ============================================================

print("Collecting Teach Mode examples...")

examples = []

for class_folder in (TEACH / "images").iterdir():

    if not class_folder.is_dir():
        continue

    class_name = class_folder.name

    for image in class_folder.iterdir():

        if not image.is_file():
            continue

        label = (
            TEACH
            / "labels"
            / class_name
            / f"{image.stem}.txt"
        )

        if not label.exists():

            print(
                f"WARNING: Missing label for {image}"
            )

            continue

        examples.append(
            (image, label, class_name)
        )

print(
    f"Found {len(examples)} Teach Mode examples."
)

# ============================================================
# SHUFFLE
# ============================================================

random.seed(42)
random.shuffle(examples)

# 80% train / 20% validation
split_index = int(len(examples) * 0.8)

train_examples = examples[:split_index]
valid_examples = examples[split_index:]

print(
    f"Teach train: {len(train_examples)}"
)

print(
    f"Teach valid: {len(valid_examples)}"
)

# ============================================================
# COPY TEACH TRAIN
# ============================================================

for image, label, class_name in train_examples:

    new_name = f"teach_{class_name}_{image.name}"

    shutil.copy2(
        image,
        OUTPUT / "train" / "images" / new_name
    )

    shutil.copy2(
        label,
        OUTPUT / "train" / "labels"
        / f"teach_{class_name}_{image.stem}.txt"
    )

# ============================================================
# COPY TEACH VALID
# ============================================================

for image, label, class_name in valid_examples:

    new_name = f"teach_{class_name}_{image.name}"

    shutil.copy2(
        image,
        OUTPUT / "valid" / "images" / new_name
    )

    shutil.copy2(
        label,
        OUTPUT / "valid" / "labels"
        / f"teach_{class_name}_{image.stem}.txt"
    )

# ============================================================
# CREATE DATA.YAML
# ============================================================

yaml_text = f"""path: {OUTPUT}

train: train/images
val: valid/images
test: test/images

nc: {len(CLASSES)}

names:
"""

for i, name in enumerate(CLASSES):

    yaml_text += f"  {i}: {name}\n"


with open(
    OUTPUT / "data.yaml",
    "w"
) as f:

    f.write(yaml_text)

print()
print("========================================")
print("Combined dataset created successfully!")
print("========================================")
print()
print(f"Location: {OUTPUT}")
print()
print("Classes:")

for i, name in enumerate(CLASSES):

    print(f"  {i}: {name}")

print()
print("Teach examples:")
print(f"  Train: {len(train_examples)}")
print(f"  Valid: {len(valid_examples)}")