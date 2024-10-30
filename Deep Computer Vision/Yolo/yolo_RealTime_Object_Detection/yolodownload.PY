# from ultralytics import YOLO
# YOLO('yolov8n.pt')  # downloads the weights automatically



import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict

class DetectionVisualizer:
    def __init__(self, confidence_threshold=0.5):
        # Load YOLOv8x model - most accurate version
        self.model = YOLO('yolov8n.pt')
        self.conf_threshold = confidence_threshold
        self.class_counts = defaultdict(int)
        self.colors = {}
        
    def generate_color(self, class_name):
        """Generate a unique color for each class"""
        if class_name not in self.colors:
            self.colors[class_name] = tuple(np.random.randint(0, 255, 3).tolist())
        return self.colors[class_name]

    def draw_detection_info(self, frame, detections):
        """Draw detailed detection information on frame"""
        # Reset class counts
        self.class_counts.clear()
        
        # Process each detection
        for detection in detections:
            boxes = detection.boxes
            for box in boxes:
                # Get box coordinates and info
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                
                # Update class count
                self.class_counts[class_name] += 1
                
                # Get color for this class
                color = self.generate_color(class_name)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw filled rectangle for text background
                text = f'{class_name} {confidence:.2f}'
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
                
                # Draw text
                cv2.putText(frame, text, (x1 + 5, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw class summary on the right side
        y_offset = 30
        cv2.putText(frame, 'Detection Summary:', (frame.shape[1] - 250, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        for class_name, count in sorted(self.class_counts.items()):
            color = self.colors[class_name]
            text = f'{class_name}: {count}'
            cv2.putText(frame, text, (frame.shape[1] - 250, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30
        
        return frame

def main():
    # Initialize visualizer
    visualizer = DetectionVisualizer(confidence_threshold=0.3)  # Lower confidence threshold for more detections
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Full HD resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Initialize FPS calculation
    fps_start_time = time.time()
    fps = 0
    frame_count = 0
    
    print("Starting detection... Press 'q' to quit")
    print(f"Available classes: {', '.join(visualizer.model.names.values())}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Update FPS
        frame_count += 1
        if frame_count >= 30:
            fps = frame_count / (time.time() - fps_start_time)
            fps_start_time = time.time()
            frame_count = 0
            
        # Run YOLOv8x inference
        results = visualizer.model(frame, conf=visualizer.conf_threshold)
        
        # Draw detections and info
        frame = visualizer.draw_detection_info(frame, results)
        
        # Add FPS counter
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('YOLOv8x Multi-Class Detection', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        # Optional: Print detailed detection info to console
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = visualizer.model.names[class_id]
                print(f"Detected {class_name} with confidence {confidence:.2f}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()