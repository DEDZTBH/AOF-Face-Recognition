from os import path

import face_recognition
import cv2

from src.knn import predict
from src.util.util import load
import time

frame_scale = 0.25
knn_pkl_name = 'knn_1719_preprocess_test_0_100_2'
webcam = 0
skip_frame = False
confirm_frames = 10

confirm_frames = int(confirm_frames / (2 if skip_frame else 1))

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(webcam)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
confirming = {}
confirmed = []
process_this_frame = True

knn_model = load(knn_pkl_name, folder=path.join('pkl', 'knn'))


def confirm_op(names):
    for n in [x for x in names if x != 'Unknown']:
        if n not in confirming:
            confirming[n] = 0
        confirming[n] += 1
        if confirming[n] >= confirm_frames:
            if n not in confirmed:
                confirmed.append(n)

    for k in confirming:
        if k not in names:
            confirming[k] = 0


while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    start = time.time()
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=frame_scale, fy=frame_scale)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=0)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = predict(face_encodings,
                             knn_model,
                             distance_threshold=0.52,
                             n_neighbors=2,
                             print_time=False)

        confirm_op(face_names)
        # print(confirming)
        # print(confirmed)

    if skip_frame:
        process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled
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

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print('Frame takes {:2f}ms'.format((time.time() - start) * 1000))

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
