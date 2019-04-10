import math
from sklearn import neighbors
from PIL import ImageDraw, ImageFont
import random
import time


font = ImageFont.truetype('fonts/Arial Bold.ttf', 12)


def knn_generate(X, y, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    start = time.time()
    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    print('Trained KNN model in {}s'.format(time.time() - start))
    return knn_clf


def predict(X_encodings, knn_clf=None, distance_threshold=0.6, n_neighbors=1):
    if knn_clf is None:
        raise Exception("Must supply knn classifier")

    # If no faces are found in the image, return an empty result.
    if len(X_encodings) == 0:
        return []

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(X_encodings, n_neighbors=n_neighbors)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_encodings))]

    # Predict classes and remove classifications that aren't within the threshold
    return [pred if rec else "Unknown" for pred, rec in
            zip(knn_clf.predict(X_encodings), are_matches)]


def show_prediction_labels_on_image(X_img, face_locations, predictions):
    """
    Shows the face recognition results visually.
    :param X_img: pil image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    draw = ImageDraw.Draw(X_img)

    # Loop through each face found in the unknown image
    for (top, right, bottom, left), name in zip(face_locations, predictions):

        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        draw.text((left + 6, bottom + random.randint(0, 20)), str(name),
                  fill=(0, 210, 0, 255), font=font)

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    X_img.show()


# if __name__ == "__main__":
#     # STEP 1: Train the KNN classifier and save it to disk
#     # Once the model is trained and saved, you can skip this step next time.
#     print("Training KNN classifier...")
#     classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
#     print("Training complete!")
#
#     # STEP 2: Using the trained classifier, make predictions for unknown images
#     for image_file in os.listdir("knn_examples/test"):
#         full_file_path = os.path.join("knn_examples/test", image_file)
#
#         print("Looking for faces in {}".format(image_file))
#
#         # Find all people in the image using a trained classifier model
#         # Note: You can pass in either a classifier file name or a classifier model instance
#         predictions = predict(full_file_path, model_path="trained_knn_model.clf")
#
#         # Print results on the console
#         for name, (top, right, bottom, left) in predictions:
#             print("- Found {} at ({}, {})".format(name, left, top))
#
#         # Display results overlaid on an image
#         show_prediction_labels_on_image(os.path.join("knn_examples/test", image_file), predictions)
