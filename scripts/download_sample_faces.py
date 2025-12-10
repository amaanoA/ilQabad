#!/usr/bin/env python3
"""Download or prepare sample face images for testing the attendance pipeline.

This script prepares sample face images for testing:
1. If data/real_faces/ exists, uses real face images
2. Otherwise, generates synthetic face images

Usage:
    python scripts/download_sample_faces.py
"""

import shutil
from pathlib import Path

import cv2
import numpy as np

# Directory paths
REAL_FACES_DIR = Path("data/real_faces")
SAMPLE_FACES_DIR = Path("data/sample_faces")


def create_synthetic_face(
    seed: int,
    width: int = 224,
    height: int = 224,
    skin_tone: tuple = (200, 160, 140),
) -> np.ndarray:
    """Create a synthetic face-like image for testing.

    This creates a simple oval pattern with skin tones that can be used
    for basic pipeline testing when real face images aren't available.

    Args:
        seed: Random seed for reproducibility
        width: Image width
        height: Image height
        skin_tone: RGB tuple for base skin color

    Returns:
        RGB image as numpy array
    """
    rng = np.random.default_rng(seed)

    # Create base image with skin tone
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = skin_tone

    # Create face oval
    center_y, center_x = height // 2, width // 2
    for y in range(height):
        for x in range(width):
            # Oval equation
            dist = ((x - center_x) / (width * 0.35)) ** 2 + (
                (y - center_y) / (height * 0.45)
            ) ** 2
            if dist < 1:
                # Inside face
                noise = rng.integers(-15, 15, 3)
                base = np.array(skin_tone)
                image[y, x] = np.clip(base + noise, 0, 255).astype(np.uint8)
            else:
                # Background
                image[y, x] = [50, 50, 50]  # Dark gray background

    # Add simple eye regions (darker circles)
    eye_y = center_y - height // 8
    left_eye_x = center_x - width // 6
    right_eye_x = center_x + width // 6
    eye_radius = width // 20

    for ey, ex in [(eye_y, left_eye_x), (eye_y, right_eye_x)]:
        for y in range(max(0, ey - eye_radius), min(height, ey + eye_radius)):
            for x in range(max(0, ex - eye_radius), min(width, ex + eye_radius)):
                if (x - ex) ** 2 + (y - ey) ** 2 < eye_radius**2:
                    image[y, x] = [80, 60, 50]  # Dark eye color

    # Add mouth region (slightly darker line)
    mouth_y = center_y + height // 6
    mouth_width = width // 6
    for x in range(center_x - mouth_width, center_x + mouth_width):
        if 0 <= x < width:
            image[mouth_y, x] = [180, 120, 110]

    return image


def create_spoof_image(real_image: np.ndarray, seed: int = 42) -> np.ndarray:
    """Create a simulated spoof (printed photo) from a real image.

    Adds artifacts typical of printed/screen photos:
    - Reduced color depth
    - Added noise (print texture)
    - Slight blur
    - Moire pattern simulation

    Args:
        real_image: Original face image (RGB)
        seed: Random seed

    Returns:
        Spoofed version of the image (RGB)
    """
    rng = np.random.default_rng(seed)
    spoof = real_image.copy().astype(np.float32)

    # Reduce color depth (quantization)
    spoof = (spoof // 32) * 32

    # Add print texture noise
    noise = rng.integers(-20, 20, spoof.shape).astype(np.float32)
    spoof = spoof + noise

    # Add slight color shift (common in prints)
    spoof[:, :, 0] = spoof[:, :, 0] * 0.95  # Reduce red slightly
    spoof[:, :, 2] = spoof[:, :, 2] * 1.05  # Increase blue slightly

    # Add subtle grid pattern (moire simulation)
    for y in range(0, spoof.shape[0], 4):
        spoof[y, :, :] = spoof[y, :, :] * 0.98
    for x in range(0, spoof.shape[1], 4):
        spoof[:, x, :] = spoof[:, x, :] * 0.98

    return np.clip(spoof, 0, 255).astype(np.uint8)


def load_image_rgb(path: Path) -> np.ndarray | None:
    """Load an image and convert to RGB format.

    Args:
        path: Path to image file

    Returns:
        RGB image as numpy array, or None if loading failed
    """
    if not path.exists():
        return None
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_image_rgb(image: np.ndarray, path: Path) -> bool:
    """Save an RGB image to file.

    Args:
        image: RGB image as numpy array
        path: Destination path

    Returns:
        True if saved successfully
    """
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(str(path), bgr)


def use_real_faces() -> dict[str, list[Path]]:
    """Use real face images from data/real_faces/.

    Maps:
    - real_face-01.jpg -> student_001
    - real_face-02.jpg -> student_002
    - real_face-03.jpg -> student_003
    - real_face-04.jpg -> unknown

    Returns:
        Dictionary mapping person IDs to list of image paths
    """
    print("Using real face images from data/real_faces/")
    print()

    # Clean up existing sample_faces directory
    if SAMPLE_FACES_DIR.exists():
        shutil.rmtree(SAMPLE_FACES_DIR)
    SAMPLE_FACES_DIR.mkdir(parents=True, exist_ok=True)

    persons = {}

    # Map real faces to students
    student_mapping = [
        ("real_face-01.jpg", "student_001"),
        ("real_face-02.jpg", "student_002"),
        ("real_face-03.jpg", "student_003"),
    ]

    print("Copying real faces to student directories...")
    for real_name, student_id in student_mapping:
        real_path = REAL_FACES_DIR / real_name
        if not real_path.exists():
            print(f"  [SKIP] {real_name} not found")
            continue

        # Create student directory
        student_dir = SAMPLE_FACES_DIR / student_id
        student_dir.mkdir(exist_ok=True)

        # Copy the image
        dest_path = student_dir / "face_1.jpg"
        shutil.copy2(real_path, dest_path)

        persons[student_id] = [dest_path]
        print(f"  [OK] {real_name} -> {student_id}/face_1.jpg")

    # Generate spoof samples from real faces
    spoof_dir = SAMPLE_FACES_DIR / "spoofs"
    spoof_dir.mkdir(exist_ok=True)

    print()
    print("Generating spoof samples from real faces...")
    for real_name, student_id in student_mapping[:2]:  # First 2 students
        real_path = REAL_FACES_DIR / real_name
        if not real_path.exists():
            continue

        # Load the real image
        image = load_image_rgb(real_path)
        if image is None:
            continue

        # Create spoof version
        spoof = create_spoof_image(image, seed=hash(student_id) % 10000)

        # Save spoof
        spoof_path = spoof_dir / f"spoof_{student_id}.jpg"
        save_image_rgb(spoof, spoof_path)
        print(f"  [OK] Created spoof from {real_name} -> spoofs/spoof_{student_id}.jpg")

    # Use real_face-04.jpg as unknown person
    unknown_dir = SAMPLE_FACES_DIR / "unknown"
    unknown_dir.mkdir(exist_ok=True)

    print()
    print("Setting up unknown person samples...")
    unknown_source = REAL_FACES_DIR / "real_face-04.jpg"
    if unknown_source.exists():
        unknown_dest = unknown_dir / "unknown_1.jpg"
        shutil.copy2(unknown_source, unknown_dest)
        print(f"  [OK] real_face-04.jpg -> unknown/unknown_1.jpg")
    else:
        print("  [SKIP] real_face-04.jpg not found")

    return persons


def generate_synthetic_faces() -> dict[str, list[Path]]:
    """Generate synthetic face samples.

    Returns:
        Dictionary mapping person IDs to list of image paths
    """
    print("Generating synthetic face samples...")
    print("(Real faces not found. For better testing, add images to data/real_faces/)")
    print()

    SAMPLE_FACES_DIR.mkdir(parents=True, exist_ok=True)

    # Skin tones for diversity
    skin_tones = [
        (210, 170, 150),  # Light
        (180, 130, 100),  # Medium
        (140, 100, 80),   # Dark
    ]

    persons = {}

    for i, person_id in enumerate(["student_001", "student_002", "student_003"]):
        person_dir = SAMPLE_FACES_DIR / person_id
        person_dir.mkdir(exist_ok=True)
        persons[person_id] = []

        skin_tone = skin_tones[i % len(skin_tones)]

        # Generate 3 variations per person
        for j in range(3):
            # Slightly vary the skin tone for each photo
            varied_tone = tuple(
                max(0, min(255, c + (j - 1) * 10)) for c in skin_tone
            )
            seed = i * 1000 + j

            image = create_synthetic_face(seed=seed, skin_tone=varied_tone)

            # Save image
            image_path = person_dir / f"face_{j+1}.png"
            save_image_rgb(image, image_path)

            persons[person_id].append(image_path)
            print(f"  Created: {image_path}")

    # Generate spoof samples
    spoof_dir = SAMPLE_FACES_DIR / "spoofs"
    spoof_dir.mkdir(exist_ok=True)

    print()
    print("Generating spoof samples...")
    for i, person_id in enumerate(["student_001", "student_002"]):
        # Create spoof from first image of each person
        original = create_synthetic_face(seed=i * 1000, skin_tone=skin_tones[i])
        spoof = create_spoof_image(original, seed=i)

        spoof_path = spoof_dir / f"spoof_{person_id}.png"
        save_image_rgb(spoof, spoof_path)
        print(f"  Created: {spoof_path}")

    # Generate unknown person samples
    unknown_dir = SAMPLE_FACES_DIR / "unknown"
    unknown_dir.mkdir(exist_ok=True)

    print()
    print("Generating unknown person samples...")
    for i in range(2):
        seed = 9000 + i
        # Use different skin tone than enrolled students
        skin_tone = (190, 150, 120)
        image = create_synthetic_face(seed=seed, skin_tone=skin_tone)

        unknown_path = unknown_dir / f"unknown_{i+1}.png"
        save_image_rgb(image, unknown_path)
        print(f"  Created: {unknown_path}")

    return persons


def main():
    """Main function to prepare sample faces."""
    print("=" * 60)
    print("Sample Face Preparation for ilQabad Attendance System")
    print("=" * 60)
    print()

    # Check if real faces are available
    real_faces_available = (
        REAL_FACES_DIR.exists()
        and len(list(REAL_FACES_DIR.glob("*.jpg"))) >= 3
    )

    if real_faces_available:
        persons = use_real_faces()
        face_type = "real"
    else:
        persons = generate_synthetic_faces()
        face_type = "synthetic"

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Sample faces directory: {SAMPLE_FACES_DIR.absolute()}")
    print(f"Face type: {face_type}")
    print(f"Students prepared: {len(persons)}")
    for person_id, paths in persons.items():
        print(f"  {person_id}: {len(paths)} image(s)")
    print()
    print("Additional samples:")
    print(f"  Spoofs: {SAMPLE_FACES_DIR / 'spoofs'}")
    print(f"  Unknown: {SAMPLE_FACES_DIR / 'unknown'}")

    if face_type == "real":
        print()
        print("Using real face images for accurate pipeline testing.")
    else:
        print()
        print("Note: These are synthetic faces for pipeline testing only.")
        print("For real accuracy testing, add face images to data/real_faces/")


if __name__ == "__main__":
    main()
