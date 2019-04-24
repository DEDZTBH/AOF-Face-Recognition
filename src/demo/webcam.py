from os import path

import face_recognition
import cv2

# Get a reference to webcam #0 (the default one)
from src.knn import predict
from src.util.util import load

frame_scale = 1
knn_pkl_name = 'knn_1719_preprocess_test_0_100_2'
webcam = 'http://172.16.5.57:8080/video'
skip_frame = False

video_capture = cv2.VideoCapture(webcam)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

knn_model = load(knn_pkl_name, folder=path.join('pkl', 'knn'))

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=frame_scale, fy=frame_scale)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = predict(face_encodings,
                             knn_model,
                             distance_threshold=0.46,
                             n_neighbors=2,
                             print_time=False)
    if skip_frame:
        process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled
        top = round(top / frame_scale)
        right = round(right / frame_scale)
        bottom = round(bottom / frame_scale)
        left = round(left / frame_scale)

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
