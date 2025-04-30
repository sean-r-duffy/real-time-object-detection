"""
Nick Cantalupa and Sean Duffy
Computer Vision Spring 2025

Main script for running the object detection system.
See README.md for instructions on how to run the program.
"""

from object_detection.classic import Frame, Database
import cv2
import argparse
import json
import object_detection.deep_learning as deep_learning
from datetime import datetime
from object_detection.utils import get_name, get_patch_func, load_adversarial_patches
import time

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)
FRAME_INTERVAL = config['FRAME_INTERVAL']


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Object Detection")
    parser.add_argument('--classic', action='store_true', help="Use classic OpenCV methods for object detection")
    parser.add_argument('--deep_learning', action='store_true', help="Use deep learning methods for object detection")
    parser.add_argument('--video_source', type=int, default=0,
                        help="Video source (system dependent - usually 0 for webcam)")
    parser.add_argument('--show_processing', action='store_true', help="Show feeds with intermediate processing steps")
    parser.add_argument('--model', type=str, help="Deep learning model to use (resnet or yolos)")
    args = parser.parse_args()

    # Open video capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video device")
        return -1

    # Create windows for displaying images
    cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
    if args.show_processing:
        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Binary", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Morphed", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Components", cv2.WINDOW_NORMAL)

    # State of the program
    state = {
        'method': 'deep_learning' if args.deep_learning else 'classic',
        'mode': 'detection',
        'patch': None,
        'patch_coords': (None, None),
        'patch_scale': 1.0,
    }

    # Initialize the database
    database = Database()

    # Initialize the deep learning model
    model = args.model if args.model else "resnet"
    network = deep_learning.DeepNetwork(model)

    start_time = time.time()
    total_frames = 0

    # Process frames until user quits
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            return -1

        # Add patch to frame if necessary
        patches = load_adversarial_patches()
        if state['patch'] is not None:
            patch = patches[state['patch']]
            if patch is not None and state['patch_coords'][0] is not None and state['patch_coords'][1] is not None:
                scale = state['patch_scale']
                h, w = patch.shape[:2]
                resized_patch = cv2.resize(patch, (int(w * scale), int(h * scale)))
                x, y = state['patch_coords']
                ph, pw = resized_patch.shape[:2]
                if y + ph <= frame.shape[0] and x + pw <= frame.shape[1]:
                    frame[y:y + ph, x:x + pw] = resized_patch

        # Process frame
        if state['method'] == 'classic':
            frame = Frame(frame)
            frame.calculate_features()
            frame.match(database)
            obj_detect_image = frame.show(database, training=(state['mode'] == 'training'))
        else:
            # Define either yolos or resnet model
            obj_detect_image = network.process_frame(frame)

        # Display
        cv2.imshow("Object Detection", obj_detect_image)
        if args.show_processing:
            cv2.imshow("Original", frame.original)
            cv2.imshow("Binary", frame.threshold)
            cv2.imshow("Morphed", frame.morphed)
            cv2.imshow("Components", frame.connected_components.image)

        # Check for mouse
        cv2.setMouseCallback('Object Detection', get_patch_func(state_dict=state, patches=patches))

        # Check for key press
        key = cv2.waitKey(FRAME_INTERVAL)
        if key == ord('q'):  # Quit program
            database.save()
            deep_learning.calcuate_fps(start_time, total_frames)
            break
        elif key == ord('s'):  # Save screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"screenshots/screenshot_{timestamp}.png", frame.image)
        elif key == ord('t'):  # Toggle between detection and training mode
            if state['mode'] == 'detection':
                state['mode'] = 'training'
                print("Switched to train mode")
            else:
                state['mode'] = 'detection'
                print("Switched to detection mode")
        elif key == ord('m'):  # Toggle between classic and deep learning methods
            if state['method'] == 'classic':
                state['method'] = 'deep_leaning'
                print("Switched to deep learning method")
            else:
                state['method'] = 'classic'
                print("Switched to classic method")
        elif key == ord('a'):  # Add new object to database
            label, name = get_name()
            if label is not None and name:
                obj = frame.label_new_object(label, name)
                if obj is not None:
                    database.add(obj)
                    print(f"Added new object: {name}")
                else:
                    print("Failed to add new object")
        elif key == ord('-'):  # Decrease patch scale
            state['patch_scale'] = max(0.1, state['patch_scale'] * 0.9)
        elif key == ord('+'):  # Increase patch scale
            state['patch_scale'] = state['patch_scale'] * 1.1

        total_frames += 1

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
