import time
from os import path

import cv2
import face_recognition

# For windows
import os
import sys
sys.path.append(os.getcwd())

from knn import predict
from util.general import load

frame_scale = 1
knn_pkl_name = 'knn_1719_preprocess_0_100_neq_2'
webcam = 0
skip_frame = True
confirm_time_ms = 1500

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(webcam)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
confirming = {}
confirmed = []
process_this_frame = True

knn_model = load(knn_pkl_name, folder=path.join('data', 'model', 'knn'))


def confirm_op(names):
    current_time = time.time()
    for n in [x for x in names if x != 'Unknown']:
        if n not in confirming:
            confirming[n] = current_time
        if (current_time - confirming[n]) >= confirm_time_ms:
            if n not in confirmed:
                confirmed.append(n)

    for k in confirming:
        if k not in names:
            confirming[k] = current_time


while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    start = time.time()
    # Resize frame of video to xxx size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=frame_scale, fy=frame_scale)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame, model='cnn')
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = predict(face_encodings,
                             knn_model,
                             distance_threshold=0.52,
                             n_neighbors=2,
                             print_time=False)

        confirm_op(face_names)
        # print(confirming)
        # print(confirmed)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to xxx
        top = round(top / frame_scale)
        right = round(right / frame_scale)
        bottom = round(bottom / frame_scale)
        left = round(left / frame_scale)

        # frame = small_frame

        color = (0, 255, 0) if name in confirmed else (0, 0, 255)

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    if skip_frame:
        process_this_frame = not process_this_frame

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # print('Frame takes {:2f}ms'.format((time.time() - start) * 1000))

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
