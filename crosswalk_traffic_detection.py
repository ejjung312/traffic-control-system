import cv2
import torch

from ultralytics import YOLO

img_size = 640
conf_threshold = 0.5
iou_threshold = 0.45
max_detection = 1000
classes = None
agnostic_nms = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# MODEL_PATH = 'model/crosswalk_best.pt'
MODEL_PATH = 'model/crosswalk_last.pt'
model = YOLO(MODEL_PATH)

# Open the video file
video_path = "datasets/sample.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(60) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()














