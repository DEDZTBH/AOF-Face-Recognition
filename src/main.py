import face_recognition
from PIL import Image, ImageDraw
import os
import math
import pickle


def get_encodings(test):
    # Create arrays of known face encodings and their names
    known_face_encodings = []
    known_face_names = []

    if test:
        directory = os.fsencode("known")
        for filename in os.listdir(directory):
            p = filename.decode("utf-8")
            print(p)
            known_face_names.append(p[0])
            known_face_encodings.append(face_recognition.face_encodings(
                face_recognition.load_image_file(os.path.join(directory, filename).decode("utf-8")))[0])
    else:
        directory = os.fsencode("2017 photos")
        for filename in os.listdir(directory):
            # p = re.compile('([A-Za-z]+\.[A-Za-z]+)\.')
            # known_face_names.append(p.match(filename.decode("utf-8")).group())
            p = filename.decode("utf-8").split("-")[0].split("-")[0]
            print(p)
            known_face_names.append(p)
            known_face_encodings.append(face_recognition.face_encodings(
                face_recognition.load_image_file(os.path.join(directory, filename).decode("utf-8")), num_jitters=100)[0])

    return known_face_encodings, known_face_names


recover = True

if recover:
    with open('2017.obj', 'rb') as file:
        saved = pickle.load(file)
        (known_face_encodings, known_face_names) = saved
else:
    known_face_encodings, known_face_names = get_encodings(0)


def face_distance_to_conf(face_distance, face_match_threshold):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))


def predict(filename, tolerance, known_face_encodings, known_face_names, showimg):
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    # Load an image with an unknown face
    unknown_image = face_recognition.load_image_file(filename)

    # Find all the faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    if showimg:
        # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
        # See http://pillow.readthedocs.io/ for more about PIL/Pillow
        pil_image = Image.fromarray(unknown_image)
        # Create a Pillow ImageDraw Draw instance to draw with
        draw = ImageDraw.Draw(pil_image)

        # Loop through each face found in the unknown image
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            face_dis = face_recognition.face_distance(known_face_encodings, face_encoding)
            matches = list(face_dis <= tolerance)

            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Draw a box around the face using the Pillow module
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

            # Draw a label with a name below the face
            text_width, text_height = draw.textsize(name)
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))

            draw.text((left + 6, bottom - text_height - 5), name +
                      "@{:.2}".format(face_distance_to_conf(face_dis.min(), tolerance)),
                      fill=(255, 255, 255, 255))

        # Remove the drawing library from memory as per the Pillow docs
        del draw

        # Display the resulting image
        pil_image.show()


predict("unknown/51341390_10156668527250236_6458268350773460992_o.jpg", 0.54, known_face_encodings, known_face_names,
        True)
