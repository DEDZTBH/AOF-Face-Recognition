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

frame_scale = 0.65
knn_pkl_name = 'knn_1719_preprocess_0_100_neq_2'

# Open the input movie file
input_movie = cv2.VideoCapture("test.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

while not input_movie.isOpened():
    time.sleep(0.01)

width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
height = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
fps = input_movie.get(cv2.CAP_PROP_FPS)

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

knn_model = load(knn_pkl_name, folder=path.join('data', 'model', 'knn'))
frame_number = 0

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break

    # start = time.time()
    # Resize frame of video to xxx size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=frame_scale, fy=frame_scale)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame, model='cnn')
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = predict(face_encodings,
                         knn_model,
                         distance_threshold=0.5,
                         n_neighbors=2,
                         print_time=False)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to xxx
        top = round(top / frame_scale)
        right = round(right / frame_scale)
        bottom = round(bottom / frame_scale)
        left = round(left / frame_scale)

        color = (0, 0, 255)

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 20), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # # Display the resulting image
    # cv2.imshow('Video', frame)

    # # Hit 'q' on the keyboard to quit!
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

    # print('Frame takes {:2f}ms'.format((time.time() - start) * 1000))

# Release handle to the webcam
input_movie.release()
cv2.destroyAllWindows()
