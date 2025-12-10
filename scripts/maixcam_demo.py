#!/usr/bin/env python3
"""Demo script for ilQabad attendance system running on Sipeed MaixCam.

This script runs the full attendance pipeline on MaixCam hardware:
1. Loads models optimized for MaixCam
2. Captures frames from the built-in camera
3. Displays results on the MaixCam screen
4. Provides audio/visual feedback for attendance

Usage:
    # On MaixCam device:
    python scripts/maixcam_demo.py

    # With specific options:
    python scripts/maixcam_demo.py --skip-liveness --models-dir /path/to/models

Prerequisites:
    - Running on Sipeed MaixCam hardware with MaixPy
    - Model files in models/ directory
    - Pre-enrolled students (or use --enroll-dir)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_status(message: str, success: bool = True) -> None:
    """Print status message with indicator."""
    indicator = "[OK]" if success else "[!!]"
    print(f"  {indicator} {message}")


def main() -> int:
    """Run MaixCam attendance demo."""
    parser = argparse.ArgumentParser(description="ilQabad MaixCam Attendance Demo")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Path to models directory",
    )
    parser.add_argument(
        "--enroll-dir",
        type=Path,
        default=None,
        help="Directory with enrollment images (student_XXX subdirectories)",
    )
    parser.add_argument(
        "--skip-liveness",
        action="store_true",
        help="Bypass liveness check for testing",
    )
    parser.add_argument(
        "--recognition-threshold",
        type=float,
        default=0.6,
        help="Face recognition threshold (default: 0.6)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Camera framerate (default: 30)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera width (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera height (default: 480)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without display output (headless mode)",
    )
    args = parser.parse_args()

    print()
    print("=" * 50)
    print("    ilQabad MaixCam Attendance System")
    print("=" * 50)
    print()

    # -------------------------------------------------------------------------
    # Check MaixPy availability
    # -------------------------------------------------------------------------
    print("[1/5] Checking MaixPy...")
    try:
        from src.infrastructure.camera.maixcam_camera import is_maixcam_available

        if not is_maixcam_available():
            print_status("MaixPy not available - not running on MaixCam", False)
            print("  This demo requires MaixCam hardware.")
            print("  For desktop testing, use: python scripts/demo_attendance_pipeline.py")
            return 1

        print_status("MaixPy available")

        # Import MaixPy display for showing results
        from maix import display as maix_display
        from maix import image as maix_image

        display_available = not args.no_display
        if display_available:
            print_status("Display available")
        else:
            print_status("Display disabled (--no-display)")

    except ImportError as e:
        print_status(f"MaixPy import error: {e}", False)
        return 1

    # -------------------------------------------------------------------------
    # Load models
    # -------------------------------------------------------------------------
    print()
    print("[2/5] Loading ML models...")

    yunet_path = args.models_dir / "detection" / "yunet.onnx"
    mobilefacenet_path = args.models_dir / "recognition" / "mobilefacenet.onnx"
    deeppixbis_path = args.models_dir / "liveness" / "deeppixbis.onnx"

    detector = None
    recognizer = None
    liveness_checker = None

    # Load detector
    try:
        from src.infrastructure.ml.yunet_detector import YuNetDetector

        start = time.time()
        detector = YuNetDetector(model_path=yunet_path)
        elapsed = (time.time() - start) * 1000
        print_status(f"YuNet detector loaded [{elapsed:.0f}ms]")
    except FileNotFoundError:
        print_status(f"YuNet model not found: {yunet_path}", False)
        return 1
    except Exception as e:
        print_status(f"YuNet load error: {e}", False)
        return 1

    # Load recognizer
    try:
        from src.infrastructure.ml.mobilefacenet_recognizer import MobileFaceNetRecognizer

        start = time.time()
        recognizer = MobileFaceNetRecognizer(model_path=mobilefacenet_path)
        elapsed = (time.time() - start) * 1000
        print_status(f"MobileFaceNet loaded [{elapsed:.0f}ms]")
    except FileNotFoundError:
        print_status(f"MobileFaceNet model not found: {mobilefacenet_path}", False)
        return 1
    except Exception as e:
        print_status(f"MobileFaceNet load error: {e}", False)
        return 1

    # Load liveness checker
    try:
        from src.infrastructure.ml.deeppixbis_liveness import DeePixBiSLiveness

        start = time.time()
        liveness_checker = DeePixBiSLiveness(model_path=deeppixbis_path)
        elapsed = (time.time() - start) * 1000
        print_status(f"DeePixBiS liveness loaded [{elapsed:.0f}ms]")
    except FileNotFoundError:
        print_status(f"DeePixBiS model not found: {deeppixbis_path}", False)
        return 1
    except Exception as e:
        print_status(f"DeePixBiS load error: {e}", False)
        return 1

    # -------------------------------------------------------------------------
    # Create pipeline
    # -------------------------------------------------------------------------
    print()
    print("[3/5] Creating attendance pipeline...")

    liveness_threshold = 0.0 if args.skip_liveness else 0.5

    try:
        from src.domain.attendance_pipeline import DefaultAttendancePipeline

        pipeline = DefaultAttendancePipeline(
            detector=detector,
            recognizer=recognizer,
            liveness_checker=liveness_checker,
            recognition_threshold=args.recognition_threshold,
            liveness_threshold=liveness_threshold,
        )
        print_status("Pipeline created")

        if args.skip_liveness:
            print_status("Liveness check BYPASSED (--skip-liveness)")
        else:
            print_status(f"Liveness threshold: {liveness_threshold}")

    except Exception as e:
        print_status(f"Pipeline creation failed: {e}", False)
        return 1

    # -------------------------------------------------------------------------
    # Enroll students (if directory provided)
    # -------------------------------------------------------------------------
    print()
    print("[4/5] Student enrollment...")

    enrolled_count = 0

    if args.enroll_dir and args.enroll_dir.exists():
        import cv2

        student_dirs = sorted([
            d for d in args.enroll_dir.iterdir()
            if d.is_dir() and d.name.startswith("student_")
        ])

        for student_dir in student_dirs:
            student_id = student_dir.name
            photos = list(student_dir.glob("*.png")) + list(student_dir.glob("*.jpg"))

            if not photos:
                continue

            # Load photos
            loaded_photos = []
            for photo_path in photos[:3]:
                bgr = cv2.imread(str(photo_path))
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    loaded_photos.append(rgb)

            if loaded_photos:
                try:
                    success = pipeline.enroll_student(student_id, loaded_photos)
                    if success:
                        enrolled_count += 1
                        print_status(f"Enrolled {student_id} ({len(loaded_photos)} photos)")
                except Exception as e:
                    print_status(f"Failed to enroll {student_id}: {e}", False)

        print_status(f"Total enrolled: {enrolled_count} students")
    else:
        print_status("No enrollment directory specified (use --enroll-dir)")
        print_status("Pipeline ready for recognition of pre-enrolled students")

    # -------------------------------------------------------------------------
    # Start camera and run attendance loop
    # -------------------------------------------------------------------------
    print()
    print("[5/5] Starting camera...")

    try:
        from src.infrastructure.camera.maixcam_camera import MaixCamSource
        from src.core.interfaces.camera import CameraConfig

        config = CameraConfig(width=args.width, height=args.height, fps=args.fps)
        camera = MaixCamSource(config)

        # Initialize display if available
        disp = None
        if display_available:
            try:
                disp = maix_display.Display()
                print_status("Display initialized")
            except Exception as e:
                print_status(f"Display init failed: {e}", False)
                disp = None

        print_status(f"Camera: {args.width}x{args.height} @ {args.fps}fps")
        print()
        print("=" * 50)
        print("  Attendance capture started")
        print("  Press Ctrl+C to stop")
        print("=" * 50)
        print()

        # Attendance tracking
        recent_recognitions: dict[str, float] = {}  # student_id -> last_recognition_time
        cooldown_seconds = 5.0  # Don't re-announce same person within 5 seconds

        frame_count = 0
        recognition_count = 0
        last_status_time = time.time()

        with camera:
            while True:
                frame = camera.capture()
                if frame is None:
                    continue

                frame_count += 1

                # Process every frame (MaixCam is slower, process all frames)
                try:
                    results = pipeline.process_frame(frame.image)

                    for result in results:
                        student_id = result.student_id
                        now = time.time()

                        # Check cooldown
                        last_seen = recent_recognitions.get(student_id, 0)
                        if now - last_seen > cooldown_seconds:
                            recognition_count += 1
                            recent_recognitions[student_id] = now

                            # Print attendance record
                            timestamp = time.strftime("%H:%M:%S")
                            print(f"  [{timestamp}] PRESENT: {student_id} (conf={result.confidence:.2f})")

                            # Visual feedback on display
                            if disp:
                                # Draw green box around face
                                pass  # Display drawing handled below

                except Exception as e:
                    print(f"  [ERR] Processing error: {e}")

                # Update display
                if disp and frame_count % 2 == 0:  # Update display every 2 frames
                    try:
                        # Convert numpy array to MaixPy image
                        maix_img = maix_image.Image(
                            frame.image.shape[1],  # width
                            frame.image.shape[0],  # height
                            maix_image.Format.FMT_RGB888,
                        )
                        # Copy frame data
                        maix_img.from_bytes(frame.image.tobytes())

                        # Draw recognition results
                        if results:
                            for result in results:
                                if result.face_bbox:
                                    x, y, w, h = result.face_bbox
                                    # Green box for recognized
                                    maix_img.draw_rectangle(
                                        x, y, x + w, y + h,
                                        (0, 255, 0),
                                        2,
                                    )
                                    # Student ID label
                                    maix_img.draw_string(
                                        x, y - 20,
                                        result.student_id,
                                        (0, 255, 0),
                                        scale=1.5,
                                    )

                        # Show on display
                        disp.show(maix_img)
                    except Exception:
                        pass  # Ignore display errors

                # Periodic status update
                now = time.time()
                if now - last_status_time >= 10.0:
                    fps = frame_count / (now - last_status_time + 0.001)
                    print(f"  [STATUS] Frames: {frame_count}, Recognitions: {recognition_count}, FPS: {fps:.1f}")
                    last_status_time = now
                    frame_count = 0

    except KeyboardInterrupt:
        print()
        print("  Stopping...")
    except Exception as e:
        print_status(f"Camera error: {e}", False)
        return 1

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print()
    print("=" * 50)
    print("  Session Summary")
    print("=" * 50)
    print(f"  Total recognitions: {recognition_count}")
    print(f"  Unique students:    {len(recent_recognitions)}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
