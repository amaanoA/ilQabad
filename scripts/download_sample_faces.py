#!/usr/bin/env python3
"""Download sample face images for testing the attendance pipeline.

This script downloads sample face images from public sources for testing.
Images are stored in data/sample_faces/ directory.

Usage:
    python scripts/download_sample_faces.py
"""

import os
import urllib.request
from pathlib import Path

import numpy as np

# Sample face image URLs from public datasets
# Using Wikimedia Commons images (public domain / CC licensed)
SAMPLE_FACE_URLS = {
    "person_001": [
        # Using placeholder - in real usage, use actual face image URLs
        # These would be replaced with real face dataset URLs
    ],
    "person_002": [],
    "person_003": [],
}

# Directory to store sample faces
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
        real_image: Original face image
        seed: Random seed

    Returns:
        Spoofed version of the image
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


def download_or_generate_samples() -> dict[str, list[Path]]:
    """Download sample faces or generate synthetic ones.

    Returns:
        Dictionary mapping person IDs to list of image paths
    """
    SAMPLE_FACES_DIR.mkdir(parents=True, exist_ok=True)

    # Skin tones for diversity
    skin_tones = [
        (210, 170, 150),  # Light
        (180, 130, 100),  # Medium
        (140, 100, 80),   # Dark
    ]

    persons = {}

    print("Generating synthetic face samples...")
    print("(In production, use real face images from LFW or similar datasets)")

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

            try:
                import cv2
                cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            except ImportError:
                # Fallback to PIL if cv2 not available for writing
                try:
                    from PIL import Image
                    Image.fromarray(image).save(image_path)
                except ImportError:
                    # Just save as numpy file
                    np.save(str(image_path).replace(".png", ".npy"), image)
                    image_path = Path(str(image_path).replace(".png", ".npy"))

            persons[person_id].append(image_path)
            print(f"  Created: {image_path}")

    # Generate spoof samples
    spoof_dir = SAMPLE_FACES_DIR / "spoofs"
    spoof_dir.mkdir(exist_ok=True)

    print("\nGenerating spoof samples...")
    for i, person_id in enumerate(["student_001", "student_002"]):
        # Create spoof from first image of each person
        original = create_synthetic_face(seed=i * 1000, skin_tone=skin_tones[i])
        spoof = create_spoof_image(original, seed=i)

        spoof_path = spoof_dir / f"spoof_{person_id}.png"
        try:
            import cv2
            cv2.imwrite(str(spoof_path), cv2.cvtColor(spoof, cv2.COLOR_RGB2BGR))
        except ImportError:
            try:
                from PIL import Image
                Image.fromarray(spoof).save(spoof_path)
            except ImportError:
                np.save(str(spoof_path).replace(".png", ".npy"), spoof)

        print(f"  Created: {spoof_path}")

    # Generate unknown person samples
    unknown_dir = SAMPLE_FACES_DIR / "unknown"
    unknown_dir.mkdir(exist_ok=True)

    print("\nGenerating unknown person samples...")
    for i in range(2):
        seed = 9000 + i
        # Use different skin tone than enrolled students
        skin_tone = (190, 150, 120)
        image = create_synthetic_face(seed=seed, skin_tone=skin_tone)

        unknown_path = unknown_dir / f"unknown_{i+1}.png"
        try:
            import cv2
            cv2.imwrite(str(unknown_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        except ImportError:
            try:
                from PIL import Image
                Image.fromarray(image).save(unknown_path)
            except ImportError:
                np.save(str(unknown_path).replace(".png", ".npy"), image)

        print(f"  Created: {unknown_path}")

    return persons


def main():
    """Main function to download/generate sample faces."""
    print("=" * 60)
    print("Sample Face Generator for ilQabad Attendance System")
    print("=" * 60)
    print()

    persons = download_or_generate_samples()

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Sample faces directory: {SAMPLE_FACES_DIR.absolute()}")
    print(f"Students generated: {len(persons)}")
    for person_id, paths in persons.items():
        print(f"  {person_id}: {len(paths)} images")
    print()
    print("Additional samples:")
    print(f"  Spoofs: {SAMPLE_FACES_DIR / 'spoofs'}")
    print(f"  Unknown: {SAMPLE_FACES_DIR / 'unknown'}")
    print()
    print("Note: These are synthetic faces for pipeline testing only.")
    print("For real accuracy testing, use actual face datasets like LFW.")


if __name__ == "__main__":
    main()
