import cv2
import numpy as np
from ultralytics import YOLO  # For easier YOLO implementation

class SimpleDetector:
    def __init__(self):
        """Initialize detector with YOLO and face detection"""
        # Initialize YOLO model (will download automatically if not present)
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Set confidence threshold
        self.conf_threshold = 0.5
        
    def detect_objects_and_faces(self, frame):
        """Detect objects and faces in the frame"""
        # Object detection using YOLO
        results = self.yolo_model(frame, conf=self.conf_threshold)
        
        # Process YOLO results
        objects = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # get box coordinates
                conf = box.conf[0]             # confidence score
                cls = int(box.cls[0])          # class number
                name = r.names[cls]            # class name
                
                objects.append({
                    'box': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                    'confidence': float(conf),
                    'class': name
                })
        
        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return objects, faces
    
    def draw_detections(self, frame, objects, faces):
        """Draw detected objects and faces on the frame"""
        # Draw objects
        for obj in objects:
            x, y, w, h = obj['box']
            label = f"{obj['class']} {obj['confidence']:.2f}"
            
            # Draw box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Face", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Add detection counts
        cv2.putText(frame, f"Objects: {len(objects)} Faces: {len(faces)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame

def main():
    """Main function to run the detection system"""
    try:
        # Initialize detector
        detector = SimpleDetector()
        print("Detector initialized successfully")
        
        # Initialize video capture (0 for webcam)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open video capture")
        
        print("Press 'q' to quit")
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Perform detection
            objects, faces = detector.detect_objects_and_faces(frame)
            
            # Draw results
            frame = detector.draw_detections(frame, objects, faces)
            
            # Display result
            cv2.imshow('Object and Face Detection', frame)
            
            # Break loop with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        # Clean up
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()