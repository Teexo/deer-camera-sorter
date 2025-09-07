#!/usr/bin/env python3
"""
Deer Camera Image Sorter
Detects humans and animals in trail camera images using YOLOv3.
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import shutil
import urllib.request
import sys
import platform

class DeerCameraSorter:
    def __init__(self, confidence_threshold=0.5, model_dir=None):
        self.confidence_threshold = confidence_threshold
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).parent
        
        # Check if OpenCV is available
        try:
            cv2.__version__
        except ImportError:
            print("Error: OpenCV is required. Install with: pip install opencv-python")
            sys.exit(1)
        
        # Ensure model directory exists
        self.model_dir.mkdir(exist_ok=True)
        
        # Download required files if missing
        if not self.download_required_files():
            sys.exit(1)
        
        # Initialize the object detection model
        try:
            self.net = cv2.dnn.readNetFromDarknet(
                str(self.model_dir / 'yolov3.cfg'),
                str(self.model_dir / 'yolov3.weights')
            )
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            sys.exit(1)
        
        # Get output layer names
        try:
            self.layer_names = self.net.getLayerNames()
            self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except Exception as e:
            print(f"Error setting up network layers: {e}")
            sys.exit(1)
        
        # Classes to detect
        self.classes = [
            'person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'deer'
        ]
        
        # Load COCO class names
        try:
            with open(self.model_dir / 'coco.names', 'r') as f:
                self.all_classes = [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"Error loading class names: {e}")
            sys.exit(1)
        
        self.colors = np.random.uniform(0, 255, size=(len(self.all_classes), 3))
    
    def download_required_files(self):
        """Download required YOLO files if they don't exist"""
        files_to_download = {
            'yolov3.cfg': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
            'coco.names': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
        }
        
        print("Checking for required model files...")
        
        # Download config and names files
        for filename, url in files_to_download.items():
            file_path = self.model_dir / filename
            if not file_path.exists():
                print(f"Downloading {filename}...")
                try:
                    urllib.request.urlretrieve(url, str(file_path))
                    print(f"✓ Downloaded {filename}")
                except Exception as e:
                    print(f"Error downloading {filename}: {e}")
                    return False
        
        # Check for weights file
        weights_path = self.model_dir / 'yolov3.weights'
        if not weights_path.exists():
            print("\n" + "="*60)
            print("yolov3.weights not found!")
            print("This file is too large to download automatically.")
            print("\nPlease download it manually:")
            print("1. Visit: https://pjreddie.com/darknet/yolo/")
            print("2. Download yolov3.weights")
            print("3. Place it in:", self.model_dir)
            print("\nOr use these commands:")
            print("curl -O https://pjreddie.com/media/files/yolov3.weights")
            print("="*60)
            return False
        
        return True
    
    def detect_objects(self, image_path):
        """Detect objects in an image"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"  Warning: Could not read image {image_path}")
                return False, []
            
            height, width = img.shape[:2]
            
            # Create blob from image
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)
            
            # Process detections
            class_ids = []
            confidences = []
            boxes = []
            
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > self.confidence_threshold:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply non-maximum suppression
            if boxes:
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)
            else:
                indexes = []
            
            detected_objects = []
            if len(indexes) > 0:
                for i in indexes.flatten():
                    class_name = self.all_classes[class_ids[i]]
                    if class_name in self.classes:
                        detected_objects.append({
                            'class': class_name,
                            'confidence': confidences[i],
                            'box': boxes[i]
                        })
            
            return len(detected_objects) > 0, detected_objects
            
        except Exception as e:
            print(f"  Error processing {image_path}: {e}")
            return False, []
    
    def process_directory(self, input_dir, output_dir, create_subfolders=True):
        """Process all images in a directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            print(f"Error: Input directory '{input_dir}' does not exist")
            return
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        if create_subfolders:
            human_dir = output_path / "humans"
            animal_dir = output_path / "animals"
            human_dir.mkdir(exist_ok=True)
            animal_dir.mkdir(exist_ok=True)
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        # Get image files
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions and f.is_file()]
        
        if not image_files:
            print(f"No image files found in {input_dir}")
            return
        
        processed = 0
        detected = 0
        
        print(f"Processing {len(image_files)} images in: {input_dir}")
        print(f"Output directory: {output_dir}")
        print("-" * 60)
        
        for image_file in image_files:
            processed += 1
            print(f"Processing ({processed}/{len(image_files)}): {image_file.name}")
            
            has_detection, objects = self.detect_objects(image_file)
            
            if has_detection:
                detected += 1
                object_names = [obj['class'] for obj in objects]
                print(f"  ✓ Detected: {', '.join(object_names)}")
                
                # Determine destination
                human_detected = any(obj['class'] == 'person' for obj in objects)
                
                if create_subfolders:
                    dest_dir = output_path / ("humans" if human_detected else "animals")
                else:
                    dest_dir = output_path
                
                # Copy file to destination
                dest_dir.mkdir(exist_ok=True)
                dest_file = dest_dir / image_file.name
                
                try:
                    shutil.copy2(image_file, dest_file)
                except Exception as e:
                    print(f"  Error copying file: {e}")
            else:
                print("  ✗ No relevant objects detected")
        
        print("-" * 60)
        print(f"Processing complete!")
        print(f"Total images processed: {processed}")
        print(f"Images with detections: {detected}")
        if processed > 0:
            print(f"Detection rate: {(detected/processed*100):.1f}%")

def main():
    parser = argparse.ArgumentParser(
        description='Sort trail camera images by detected humans and animals using YOLOv3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python deer_camera_sorter.py ./photos ./sorted
  python deer_camera_sorter.py ./camera_images ./output --confidence 0.6
  python deer_camera_sorter.py ./input ./output --no-subfolders --model-dir ./models
        """
    )
    
    parser.add_argument('input_dir', help='Input directory containing images')
    parser.add_argument('output_dir', help='Output directory for sorted images')
    parser.add_argument('--confidence', type=float, default=0.5, 
                       help='Confidence threshold (0.0-1.0), default: 0.5')
    parser.add_argument('--model-dir', help='Directory containing YOLO model files')
    parser.add_argument('--no-subfolders', action='store_true', 
                       help='Do not create human/animal subfolders')
    
    args = parser.parse_args()
    
    # Create sorter instance
    sorter = DeerCameraSorter(
        confidence_threshold=args.confidence,
        model_dir=args.model_dir
    )
    
    # Process the directory
    sorter.process_directory(
        args.input_dir,
        args.output_dir,
        create_subfolders=not args.no_subfolders
    )

if __name__ == "__main__":
    main()
