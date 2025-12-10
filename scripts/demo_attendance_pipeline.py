#!/usr/bin/env python3
"""Demo script to test the full attendance pipeline end-to-end.

This script demonstrates the complete attendance processing flow:
1. Load all ML components (YuNet, MobileFaceNet, DeePixBiS)
2. Enroll test students using synthetic face images
3. Test recognition with known faces, unknown faces, and spoofs
4. Display formatted results with timing information

Usage:
    python scripts/demo_attendance_pipeline.py

    # With webcam capture:
    python scripts/demo_attendance_pipeline.py --webcam

Prerequisites:
    - Run `python scripts/download_sample_faces.py` first to generate test images
    - Ensure model files are in place (models/detection/, models/recognition/, models/liveness/)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def format_size(path: Path) -> str:
    """Format file size in human-readable format."""
    if not path.exists():
        return "N/A"
    size = path.stat().st_size
    if size < 1024:
        return f"{size}B"
    elif size < 1024 * 1024:
        return f"{size // 1024}KB"
    else:
        return f"{size / (1024 * 1024):.1f}MB"


def load_image(path: Path) -> np.ndarray | None:
    """Load image from path and convert to RGB."""
    if not path.exists():
        return None
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def print_header(title: str) -> None:
    """Print a section header."""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_result(label: str, success: bool, details: str = "") -> None:
    """Print a result line with status indicator."""
    status = "[PASS]" if success else "[FAIL]"
    if details:
        print(f"  {status} {label}: {details}")
    else:
        print(f"  {status} {label}")


def main() -> int:
    """Run the attendance pipeline demo."""
    parser = argparse.ArgumentParser(description="ilQabad Attendance Pipeline Demo")
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Use webcam for live capture demo",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Path to models directory",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("data/sample_faces"),
        help="Path to sample faces directory",
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("       ilQabad Attendance Pipeline Demo")
    print("=" * 60)
    print()

    # Check sample faces exist
    if not args.samples_dir.exists():
        print(f"Sample faces not found at {args.samples_dir}")
        print("Run: python scripts/download_sample_faces.py")
        return 1

    # Model paths
    yunet_path = args.models_dir / "detection" / "yunet.onnx"
    mobilefacenet_path = args.models_dir / "recognition" / "mobilefacenet.onnx"
    deeppixbis_path = args.models_dir / "liveness" / "deeppixbis.onnx"

    # -------------------------------------------------------------------------
    # Step 1: Load Models
    # -------------------------------------------------------------------------
    print_header("Loading Models")

    models_loaded = True
    detector = None
    recognizer = None
    liveness = None

    # Load YuNet detector
    print(f"  Loading YuNet detector...")
    start = time.time()
    try:
        from src.infrastructure.ml.yunet_detector import YuNetDetector
        detector = YuNetDetector(model_path=yunet_path)
        elapsed = (time.time() - start) * 1000
        print(f"  [OK] YuNet detector loaded ({format_size(yunet_path)}) [{elapsed:.0f}ms]")
    except FileNotFoundError:
        print(f"  [SKIP] YuNet model not found: {yunet_path}")
        models_loaded = False
    except Exception as e:
        print(f"  [FAIL] YuNet failed: {e}")
        models_loaded = False

    # Load MobileFaceNet recognizer
    print(f"  Loading MobileFaceNet recognizer...")
    start = time.time()
    try:
        from src.infrastructure.ml.mobilefacenet_recognizer import MobileFaceNetRecognizer
        recognizer = MobileFaceNetRecognizer(model_path=mobilefacenet_path)
        elapsed = (time.time() - start) * 1000
        print(f"  [OK] MobileFaceNet recognizer loaded ({format_size(mobilefacenet_path)}) [{elapsed:.0f}ms]")
    except FileNotFoundError:
        print(f"  [SKIP] MobileFaceNet model not found: {mobilefacenet_path}")
        models_loaded = False
    except Exception as e:
        print(f"  [FAIL] MobileFaceNet failed: {e}")
        models_loaded = False

    # Load DeePixBiS liveness
    print(f"  Loading DeePixBiS liveness...")
    start = time.time()
    try:
        from src.infrastructure.ml.deeppixbis_liveness import DeePixBiSLiveness
        liveness = DeePixBiSLiveness(model_path=deeppixbis_path)
        elapsed = (time.time() - start) * 1000
        print(f"  [OK] DeePixBiS liveness loaded ({format_size(deeppixbis_path)}) [{elapsed:.0f}ms]")
    except FileNotFoundError:
        print(f"  [SKIP] DeePixBiS model not found: {deeppixbis_path}")
        models_loaded = False
    except Exception as e:
        print(f"  [FAIL] DeePixBiS failed: {e}")
        models_loaded = False

    if not models_loaded:
        print()
        print("Some models failed to load. Demo will run with limited functionality.")
        print("Download models to the models/ directory to enable full demo.")

    # -------------------------------------------------------------------------
    # Step 2: Create Pipeline
    # -------------------------------------------------------------------------
    pipeline = None
    if detector and recognizer and liveness:
        print_header("Creating Attendance Pipeline")
        try:
            from src.domain.attendance_pipeline import DefaultAttendancePipeline
            pipeline = DefaultAttendancePipeline(
                detector=detector,
                recognizer=recognizer,
                liveness_checker=liveness,
                recognition_threshold=0.6,
                liveness_threshold=0.5,
            )
            print("  [OK] AttendancePipeline created successfully")
        except Exception as e:
            print(f"  [FAIL] Failed to create pipeline: {e}")

    # -------------------------------------------------------------------------
    # Step 3: Enroll Students
    # -------------------------------------------------------------------------
    print_header("Enrolling Students")

    enrolled_students = []
    students_dir = args.samples_dir

    # Find student directories
    student_dirs = sorted([
        d for d in students_dir.iterdir()
        if d.is_dir() and d.name.startswith("student_")
    ])

    if not student_dirs:
        print("  No student directories found.")
        print(f"  Expected: {students_dir}/student_001/, student_002/, etc.")
    else:
        for student_dir in student_dirs:
            student_id = student_dir.name
            photos = sorted(student_dir.glob("*.png")) + sorted(student_dir.glob("*.jpg")) + sorted(student_dir.glob("*.npy"))

            if not photos:
                print(f"  [SKIP] {student_id}: No photos found")
                continue

            # Load photos
            loaded_photos = []
            for photo_path in photos[:3]:  # Use up to 3 photos
                if photo_path.suffix == ".npy":
                    img = np.load(str(photo_path))
                else:
                    img = load_image(photo_path)
                if img is not None:
                    loaded_photos.append(img)

            if not loaded_photos:
                print(f"  [SKIP] {student_id}: Could not load any photos")
                continue

            # Enroll if pipeline available
            if pipeline:
                try:
                    start = time.time()
                    success = pipeline.enroll_student(student_id, loaded_photos)
                    elapsed = (time.time() - start) * 1000
                    if success:
                        enrolled_students.append(student_id)
                        print(f"  [OK] {student_id} enrolled with {len(loaded_photos)} photo(s) [{elapsed:.0f}ms]")
                    else:
                        print(f"  [FAIL] {student_id}: Enrollment failed")
                except Exception as e:
                    print(f"  [FAIL] {student_id}: {e}")
            else:
                # Just report photos found
                print(f"  [INFO] {student_id}: {len(loaded_photos)} photo(s) found (pipeline unavailable)")
                enrolled_students.append(student_id)

    print()
    print(f"  Total enrolled: {len(enrolled_students)} student(s)")

    # -------------------------------------------------------------------------
    # Step 4: Test Recognition
    # -------------------------------------------------------------------------
    print_header("Testing Recognition")

    tests_run = 0
    tests_passed = 0

    # Test 4a: Known faces (should recognize)
    print()
    print("  Testing known faces (enrolled students):")
    for student_id in enrolled_students[:2]:  # Test first 2 enrolled
        student_dir = students_dir / student_id
        test_photo = list(student_dir.glob("*.png")) + list(student_dir.glob("*.jpg"))
        if not test_photo:
            test_photo = list(student_dir.glob("*.npy"))
        if not test_photo:
            continue

        img = load_image(test_photo[0]) if test_photo[0].suffix in (".png", ".jpg") else np.load(str(test_photo[0]))
        if img is None:
            continue

        tests_run += 1
        if pipeline:
            start = time.time()
            results = pipeline.process_frame(img)
            elapsed = (time.time() - start) * 1000

            if results and results[0].student_id == student_id:
                tests_passed += 1
                conf = results[0].confidence
                print(f"    [PASS] {student_id}: Recognized (conf={conf:.2f}) [{elapsed:.0f}ms]")
            elif results:
                print(f"    [FAIL] {student_id}: Wrong match ({results[0].student_id})")
            else:
                print(f"    [FAIL] {student_id}: Not recognized [{elapsed:.0f}ms]")
        else:
            print(f"    [SKIP] {student_id}: Pipeline unavailable")

    # Test 4b: Spoof faces (should reject)
    print()
    print("  Testing spoof faces (should be rejected):")
    spoof_dir = students_dir / "spoofs"
    if spoof_dir.exists():
        spoof_files = list(spoof_dir.glob("*.png")) + list(spoof_dir.glob("*.jpg")) + list(spoof_dir.glob("*.npy"))
        for spoof_path in spoof_files[:2]:  # Test first 2 spoofs
            img = load_image(spoof_path) if spoof_path.suffix in (".png", ".jpg") else np.load(str(spoof_path))
            if img is None:
                continue

            tests_run += 1
            if pipeline:
                start = time.time()
                results = pipeline.process_frame(img)
                elapsed = (time.time() - start) * 1000

                if not results:
                    tests_passed += 1
                    print(f"    [PASS] {spoof_path.name}: Rejected (spoof detected) [{elapsed:.0f}ms]")
                else:
                    print(f"    [FAIL] {spoof_path.name}: Should have been rejected [{elapsed:.0f}ms]")
            else:
                print(f"    [SKIP] {spoof_path.name}: Pipeline unavailable")
    else:
        print("    No spoof samples found")

    # Test 4c: Unknown faces (should not match)
    print()
    print("  Testing unknown faces (should not match enrolled):")
    unknown_dir = students_dir / "unknown"
    if unknown_dir.exists():
        unknown_files = list(unknown_dir.glob("*.png")) + list(unknown_dir.glob("*.jpg")) + list(unknown_dir.glob("*.npy"))
        for unknown_path in unknown_files[:2]:  # Test first 2 unknowns
            img = load_image(unknown_path) if unknown_path.suffix in (".png", ".jpg") else np.load(str(unknown_path))
            if img is None:
                continue

            tests_run += 1
            if pipeline:
                start = time.time()
                results = pipeline.process_frame(img)
                elapsed = (time.time() - start) * 1000

                if not results:
                    tests_passed += 1
                    print(f"    [PASS] {unknown_path.name}: Not matched (correct) [{elapsed:.0f}ms]")
                else:
                    print(f"    [FAIL] {unknown_path.name}: Should not have matched [{elapsed:.0f}ms]")
            else:
                print(f"    [SKIP] {unknown_path.name}: Pipeline unavailable")
    else:
        print("    No unknown samples found")

    # -------------------------------------------------------------------------
    # Step 5: Individual Component Tests
    # -------------------------------------------------------------------------
    print_header("Component Benchmarks")

    # Test detector performance
    if detector:
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        times = []
        for _ in range(10):
            start = time.time()
            detector.detect(test_image)
            times.append((time.time() - start) * 1000)
        avg_time = sum(times) / len(times)
        print(f"  YuNet detection:     avg {avg_time:.1f}ms (over 10 runs)")

    # Test recognizer performance
    if recognizer:
        test_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        times = []
        for _ in range(10):
            start = time.time()
            recognizer.extract(test_face)
            times.append((time.time() - start) * 1000)
        avg_time = sum(times) / len(times)
        print(f"  MobileFaceNet embed: avg {avg_time:.1f}ms (over 10 runs)")

    # Test liveness performance
    if liveness:
        test_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        times = []
        for _ in range(10):
            start = time.time()
            liveness.check(test_face)
            times.append((time.time() - start) * 1000)
        avg_time = sum(times) / len(times)
        print(f"  DeePixBiS liveness:  avg {avg_time:.1f}ms (over 10 runs)")

    # -------------------------------------------------------------------------
    # Step 6: Webcam Demo (optional)
    # -------------------------------------------------------------------------
    if args.webcam and pipeline:
        print_header("Webcam Live Demo")
        print("  Press 'q' to quit, 's' to take snapshot")
        print()

        try:
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource
            from src.core.interfaces.camera import CameraConfig

            config = CameraConfig(width=640, height=480, fps=30)
            camera = OpenCVCameraSource(config)

            with camera:
                if not camera.is_running:
                    print("  [FAIL] Could not open webcam")
                else:
                    print("  [OK] Webcam opened, starting capture...")
                    frame_count = 0
                    recognized_count = 0

                    while True:
                        frame = camera.capture()
                        if frame is None:
                            continue

                        frame_count += 1

                        # Process every 5th frame for performance
                        if frame_count % 5 == 0:
                            results = pipeline.process_frame(frame.image)
                            if results:
                                for r in results:
                                    recognized_count += 1
                                    print(f"    [{frame_count}] Recognized: {r.student_id} (conf={r.confidence:.2f})")

                        # Display frame (BGR for OpenCV display)
                        display = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
                        cv2.putText(
                            display,
                            f"Frame: {frame_count} | Recognized: {recognized_count}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                        cv2.imshow("ilQabad Attendance Demo", display)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('s'):
                            snapshot_path = f"snapshot_{frame_count}.jpg"
                            cv2.imwrite(snapshot_path, display)
                            print(f"    Saved: {snapshot_path}")

                    cv2.destroyAllWindows()
                    print()
                    print(f"  Captured {frame_count} frames, recognized {recognized_count} times")

        except ImportError as e:
            print(f"  [FAIL] Camera module error: {e}")
        except Exception as e:
            print(f"  [FAIL] Webcam error: {e}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print_header("Summary")

    print(f"  Models loaded:     {'Yes' if models_loaded else 'Partial'}")
    print(f"  Pipeline ready:    {'Yes' if pipeline else 'No'}")
    print(f"  Students enrolled: {len(enrolled_students)}")
    print(f"  Tests run:         {tests_run}")
    print(f"  Tests passed:      {tests_passed}/{tests_run}" if tests_run > 0 else "  Tests passed:      N/A")
    print()

    if tests_run > 0 and tests_passed == tests_run:
        print("  All tests passed!")
        return 0
    elif tests_run == 0:
        print("  No tests were run. Check model files and sample data.")
        return 1
    else:
        print(f"  {tests_run - tests_passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
