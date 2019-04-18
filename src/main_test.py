import time
import face_recognition
from PIL import Image, ImageDraw, ImageFont
import pickle
from src.preprocess.preprocess import get_encodings
from src.util.util import face_distance_to_conf, transform_2017_photos
import numpy as np

font = ImageFont.truetype('fonts/Arial Bold.ttf', 12)

recover = True
if recover:
    with open('pkl/saved.pkl', 'rb') as file:
        saved = pickle.load(file)
        (known_face_encodings, known_face_names) = saved
else:
    known_face_encodings, known_face_names = get_encodings('data/2017 photos',
                                                           file_name_transform=transform_2017_photos,
                                                           num_jitters=100)
    # known_face_encodings, known_face_names = get_encodings(num_jitters=100)
    with open('pkl/saved.pkl', 'wb') as file:
        pickle.dump((known_face_encodings, known_face_names), file)


def predict(filename, tolerance, known_face_encodings, known_face_names, showimg):
    # Initialize some variables
    # face_locations = []
    # face_encodings = []
    # face_names = []
    # process_this_frame = True

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
        start = time.time()
        do_test = True
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            face_dis = face_recognition.face_distance(known_face_encodings, face_encoding)

            name = 'Unknown'

            # matches = list(face_dis <= tolerance)
            #
            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            best_match_index = np.argmin(face_dis)
            if face_dis[best_match_index] <= tolerance:
                name = known_face_names[best_match_index]

            # Draw a box around the face using the Pillow module
            if not do_test:
                draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

            # Draw a label with a name below the face
            # text_width, text_height = draw.textsize(name)
            # draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
            if not do_test:
                draw.text((left + 6, bottom + np.random.randint(0, 20)), name +
                          '({:.1f}%)'.format(face_distance_to_conf(face_dis.min(), tolerance) * 100),
                          fill=(0, 210, 0, 255), font=font)

        print('Made {} predictions in {:.3f}ms'.format(len(face_encodings), (time.time() - start) * 1000))

        # Remove the drawing library from memory as per the Pillow docs
        del draw

        # Display the resulting image
        pil_image.show()


predict('data/unknown/51341390_10156668527250236_6458268350773460992_o.jpg',
        0.54,
        known_face_encodings,
        known_face_names,
        True)

# predict('data/unknown/2019PoetryFinalFour.jpg',
#         0.54,
#         known_face_encodings,
#         known_face_names,
#         True)
