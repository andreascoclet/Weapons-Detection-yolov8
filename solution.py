from typing import List
import cv2
import numpy as np
from models import YOLOMODEL

model = YOLOMODEL(
    weights_path='./weights/model.pt',
    task='detect'
)


def format_predictions(labels, scores, bboxes) -> str:
    classes = []
    for label, score, bbox in zip(labels, scores, bboxes):
        if label == 'long_weapons':
            classes.append('long_weapons')
        elif label == 'knife':
            classes.append('unclassified_weapons')
        elif label == 'short_weapons':
            classes.append('short_weapons')

    return f"{'True' if classes else 'False'};{classes}"


def draw_bounding_box(frame, bboxes, labels, scores):
    for bbox, label, score in zip(bboxes, labels, scores):
        xmin, ymin, xmax, ymax = map(int, bbox)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(frame, f"{label}: {score:.2f}", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def main():
    cap = cv2.VideoCapture(0)  # Open the default camera (usually webcam)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        bboxes, labels, scores = model.predict(frame)
        prediction_repr = format_predictions(labels, scores, bboxes)
        draw_bounding_box(frame, bboxes, labels, scores)

        # Display the frame with bounding boxes
        cv2.imshow('Object Detection', frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
