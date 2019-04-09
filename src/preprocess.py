import os
import face_recognition


def get_encodings(location="known", file_name_transform=lambda x: x, num_jitters=1):
    # Create arrays of known face encodings and their names
    known_face_encodings = []
    known_face_names = []

    directory = os.fsencode(location)
    for filename in os.listdir(directory):
        p = filename.decode("utf-8")
        known_face_names.append(file_name_transform(p))
        known_face_encodings.append(face_recognition.face_encodings(
            face_recognition.load_image_file(
                os.path.join(directory, filename).decode("utf-8")),
            num_jitters=num_jitters)[0])

    return known_face_encodings, known_face_names
