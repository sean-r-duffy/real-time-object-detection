# Object Detection Methods
### Nick Cantalupa and Sean Duffy
### CS5330 Computer Vision Final Project

This project expands and improves upon the object recognition system implemented in project 3.
The classical object detection system is now implemented in Python, runs faster, is easier to train, and is more specific.
We also implemented a deep learning based system, this makes use of a small YOLOS model and Meta's DETR ResNet50.
We compare the performance of the detection methods in light and dark environments and compare their throughput in FPS.  
We then calculate adversarial patches and attack the detection systems, comparing their performance against these attacks.
The attacks can be performed both physically and digitally.  

## Project Presentation

[Report](reports/final_report.pdf)
[Presentation](reports/presentation.pdf)
[Demo](https://drive.google.com/file/d/16zuR5_MeYEpabxJIyhwkWrJ2kQpBJ-wd/view?usp=sharing)

## Command Line Arguments

- `--classic`: Start in classic OpenCV-based object detection mode
- `--deep_learning`: Start in deep learning-based mode
- `--video_source [int]`: Select video source (default: 0 for webcam)
- `--show_processing`: Display intermediate processing steps in separate windows
- `--model [str]`: Choose deep learning model ('resnet' or 'yolos')

## Key Controls

- `q`: Quit application and save database
- `s`: Save current frame as screenshot
- `t`: Toggle between detection and training modes
- `m`: Toggle between classic and deep learning methods
- `a`: Add new object to database (training mode)
- `-`: Decrease adversarial patch size
- `+`: Increase adversarial patch size

## Mouse Controls

- Left click: Place adversarial patch at clicked location
- Right click: Remove placed adversarial patch

## Adversarial Patch Calculation

To calculate an adversarial patch:

1. Run `adversarial.py` with target ImageNet class:
   ```bash
   python adversarial.py --target [int]
   ```
2. The script will generate an adversarial patch that:
    - Is robust to small changes in scale, location, and orientation
    - Is saved as adversarial_patch.png

The generated patch can be printed or applied using mouse controls during detection.

## Modes

### Classic

#### Detection Mode

Automatically detects and identifies objects in the video feed.
Only objects that are similar to an object represented in the database will be detected and displayed.

#### Training Mode

Allows adding new objects to the database for classic method detection. Use the 'a' key to initiate object addition
process. Provide the ID associated with the bounding box and the label for that bound object.

### Deep Learning

Uses the model specified via the command line arguments to detect and classify objects.

## Processing Windows

When `--show_processing` is enabled, the following windows are displayed:

- Original: Raw camera input
- Binary: Thresholded image
- Morphed: After morphological operations
- Components: Connected components visualization


