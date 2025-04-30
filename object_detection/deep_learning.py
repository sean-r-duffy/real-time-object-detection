"""
Nick Cantalupa and Sean Duffy
Computer Vision Spring 2025

This file contains the DeepNetwork class, which is responsible for loading and using deep learning models for object detection.
It uses the Hugging Face Transformers library to load models DETR and YOLOS.
It also contains the function for calculating frames per second (FPS) for the video processing.
"""

from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import YolosImageProcessor, YolosForObjectDetection
import torch
import cv2
from PIL import Image, ImageDraw
import numpy as np
import time

class DeepNetwork:
    
    # Load model, either 'resnet' or 'yolos'
    def __init__(self, model_name: str):
        self.processor, self.model = self.load_model(model_name)
        self.model.eval()

    # Load the model and processor based on the model name
    # Returns a tuple of processor and model
    def load_model(self, model_name: str) -> tuple:

        if model_name == "resnet":
            processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
            model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        elif model_name == "yolos":
            processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
            model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
        return (processor, model)

    # Process a single frame
    # This function takes a frame (OpenCV Mat), converts it to a PIL image, and performs object detection using the model.
    # Returns the processed frame with bounding boxes and labels drawn on it.
    def process_frame(self, frame: cv2.Mat) -> cv2.Mat:
        # Convert frame (OpenCV uses BGR) to RGB for Pillow
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)


        with torch.inference_mode():
            # Resize and normalize the image, inference with model
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)

            # Post-process the outputs
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        # Draw bounding boxes and labels on the image
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            print(
                    f"Detected {self.model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                    
            )
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1]), self.model.config.id2label[label.item()], fill="green")

        # Convert back to OpenCV format (BGR)
        frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        return frame_bgr
    
# Function to calculate frames per second (FPS)
# This function calculates the frames per second (FPS) based on the elapsed time and total frames processed.
def calcuate_fps(start_time: float, total_frames: int) -> float:
    elapsed_time = time.time() - start_time
    fps = total_frames / elapsed_time if elapsed_time > 0 else 0
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    print(f"Total Frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    return fps
