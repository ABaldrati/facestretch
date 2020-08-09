import random
import sys
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
from skimage import transform
from tensorflow import keras

from utils import parse_image_path, extract_landmarks, normalize_landmarks, interpolate_landmarks, \
    normalize_landmarks_eyes


def rotate_image(image_array: np.ndarray, degree: int):
    return sk.transform.rotate(image_array, degree, preserve_range=True).astype(np.uint8)


def horizontal_flip(image_array):
    return image_array[:, ::-1]


def generate_neural_network_dataset(src_path: Path, rates: List[int], normalize_eyes=True):
    """
        This function generate a dataset intended to be used by a neural network.
        Each sample is normalized subtracting to each action image the corresponding neuter image of the subject.
        In order to augment the pair considered we interpolate the sample between the neuter images and the action images
        with rates specified by `rates` parameter. Since a neural network need a lot of data to work properly we perform
        a data augmentation flipping, rotating and cropping the images.
        In order to work this function need the dataset image files in format: `name_action.ext`

        We have used this function to generate the validation dataset since we need the same dataset for each epoch in
        order to have a fair comparison among them.

        :param src_path: input folder
        :param rates: rates used to interpolate landmarks
        :param normalize_eyes: if True normalize landmarks with respect to eyes and the nose instead of the whole landmarks' bounding box
        :return:
            * landmarks_matrix: matrix which contains all the normalized landmarks, dim: [num_samples, 68 * 2]
            * training_pairs_labels: matrix with the labels for each action, dim: [num_samples, num_actions]
      """

    NUM_CROPPING_ITERATIONS = 5

    MIN_ROTATION_DEGREE = -14
    MAX_ROTATION_DEGREE = 16
    STEP_ROTATION_DEGREE = 2

    CROPPING_TOLERANCE = 0.7

    if 0 in rates:
        rates.remove(0)
    if 1 not in rates:
        rates.append(1)

    action_images = filter(lambda i: "neutro" not in i.name, src_path.iterdir())

    landmarks_matrix = np.empty((0, 68 * 2))
    labels = np.empty((0, 8), dtype=np.int16)

    neuter_landmarks = {}

    subjects = sorted(set(map(lambda i: parse_image_path(i)[0], src_path.iterdir())))
    actions = sorted(set(map(lambda i: parse_image_path(i)[1], src_path.iterdir())))

    face_to_label_matrix = np.identity(len(actions))
    face_row_mapping = {}
    for i, action in enumerate(sorted(list(actions))):
        face_row_mapping[action] = i

    for subject in subjects:
        print(f"Importing subject {subject}")
        subject_neuter_image_path = list(src_path.glob(f"{subject}_neutro.*"))[0]
        subject_neuter_image = cv2.imread(str(subject_neuter_image_path), cv2.IMREAD_UNCHANGED)
        landmarks_subject_image = extract_landmarks(subject_neuter_image)

        if not landmarks_subject_image:
            print(f"Neuter landmarks not found for subject {subject}. Skipping...", file=sys.stderr)
            action_images = list(filter(lambda i: subject not in i.name, list(action_images)))

            continue

        landmark_min_x = np.min(landmarks_subject_image[0][:, 0])
        landmark_max_x = np.max(landmarks_subject_image[0][:, 0])
        landmark_min_y = np.min(landmarks_subject_image[0][:, 1])
        landmark_max_y = np.max(landmarks_subject_image[0][:, 1])

        if normalize_eyes:
            neuter_subject_landmarks = normalize_landmarks_eyes(landmarks_subject_image[0]).flatten()
        else:
            neuter_subject_landmarks = normalize_landmarks(landmarks_subject_image[0]).flatten()

        print(f"Extracted landmarks from {subject_neuter_image_path}")

        neuter_landmarks[subject] = neuter_subject_landmarks
        # landmarks_matrix = np.vstack([landmarks_matrix, neuter_subject_landmarks])
        for ith_cropping in range(NUM_CROPPING_ITERATIONS):
            cropping_min_x_coordinate = random.randint(0, int(CROPPING_TOLERANCE * landmark_min_x))
            cropping_max_x_coordinate = random.randint(
                subject_neuter_image.shape[1] - int(
                    CROPPING_TOLERANCE * (subject_neuter_image.shape[1] - landmark_max_x)),
                subject_neuter_image.shape[1])
            cropping_min_y_coordinate = random.randint(0, int(CROPPING_TOLERANCE * landmark_min_y))
            cropping_max_y_coordinate = random.randint(
                subject_neuter_image.shape[0] - int(
                    CROPPING_TOLERANCE * (subject_neuter_image.shape[0] - landmark_max_y)),
                subject_neuter_image.shape[0])

            cropped_image = subject_neuter_image[cropping_min_y_coordinate:cropping_max_y_coordinate,
                            cropping_min_x_coordinate: cropping_max_x_coordinate, :]

            for degree in range(MIN_ROTATION_DEGREE, MAX_ROTATION_DEGREE, STEP_ROTATION_DEGREE):
                rotated_image = rotate_image(cropped_image, degree)
                landmarks_cropped_rotated_found = extract_landmarks(rotated_image)
                if not landmarks_cropped_rotated_found:
                    print(
                        f"Neuter landmarks not found for subject {subject}. At angle {degree} at {ith_cropping} Skipping...",
                        file=sys.stderr)
                    action_images = list(filter(lambda i: subject not in i.name, list(action_images)))

                else:
                    if normalize_eyes:
                        normalized_landmarks = normalize_landmarks_eyes(landmarks_cropped_rotated_found[0]).flatten()
                    else:
                        normalized_landmarks = normalize_landmarks(landmarks_cropped_rotated_found[0]).flatten()
                    rescaled_normalized_landmarks = normalized_landmarks - neuter_landmarks[subject]
                    landmarks_matrix = np.vstack([landmarks_matrix, rescaled_normalized_landmarks])
                    labels = np.vstack([labels, face_to_label_matrix[face_row_mapping["neutro"]]])

    for image_path in sorted(action_images):
        print(f"Processing {image_path}")
        subject, action = parse_image_path(image_path)
        subject_action_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        landmarks_subject_action_image = extract_landmarks(subject_action_image)
        if not landmarks_subject_action_image:
            print(f"Action landmarks ({action}) not found for subject {subject} Skipping...",
                  file=sys.stderr)
            continue
        landmark_min_x = np.min(landmarks_subject_action_image[0][:, 0])
        landmark_max_x = np.max(landmarks_subject_action_image[0][:, 0])
        landmark_min_y = np.min(landmarks_subject_action_image[0][:, 1])
        landmark_max_y = np.max(landmarks_subject_action_image[0][:, 1])

        for ith_cropping in range(NUM_CROPPING_ITERATIONS):
            cropping_min_x_coordinate = random.randint(0, int(CROPPING_TOLERANCE * landmark_min_x))
            cropping_max_x_coordinate = random.randint(
                subject_action_image.shape[1] - int(
                    CROPPING_TOLERANCE * (subject_action_image.shape[1] - landmark_max_x)),
                subject_action_image.shape[1])
            cropping_min_y_coordinate = random.randint(0, int(CROPPING_TOLERANCE * landmark_min_y))
            cropping_max_y_coordinate = random.randint(
                subject_action_image.shape[0] - int(
                    CROPPING_TOLERANCE * (subject_action_image.shape[0] - landmark_max_y)),
                subject_action_image.shape[0])

            cropped_image = subject_action_image[cropping_min_y_coordinate:cropping_max_y_coordinate,
                            cropping_min_x_coordinate: cropping_max_x_coordinate, :]

            for degree in range(MIN_ROTATION_DEGREE, MAX_ROTATION_DEGREE, STEP_ROTATION_DEGREE):
                rotated_image = rotate_image(cropped_image, degree)
                landmarks_cropped_rotated_found = extract_landmarks(rotated_image)
                if not landmarks_cropped_rotated_found:
                    print(
                        f"Action landmarks ({action}) not found for subject {subject}. At angle {degree} at {ith_cropping} Skipping...",
                        file=sys.stderr)
                else:
                    if normalize_eyes:
                        action_subject_landmarks = normalize_landmarks_eyes(
                            landmarks_cropped_rotated_found[0]).flatten()
                    else:
                        action_subject_landmarks = normalize_landmarks(landmarks_cropped_rotated_found[0]).flatten()

                    neuter_subject_landmarks = neuter_landmarks[subject]

                    for rate in rates:
                        interpolated_subject_landmarks = interpolate_landmarks(neuter_subject_landmarks,
                                                                               action_subject_landmarks,
                                                                               rate) - neuter_subject_landmarks
                        landmarks_matrix = np.vstack([landmarks_matrix, interpolated_subject_landmarks])
                        labels = np.vstack([labels, rate * face_to_label_matrix[face_row_mapping[action]] +
                                            (1 - rate) * face_to_label_matrix[face_row_mapping["neutro"]]])

    return landmarks_matrix, labels


def dataset_generator(src_path: Path, batch_size: int, normalize_eyes=True):
    """
        This function is a generator which each step create a batch of data for a neural network training.
        Each sample is normalized subtracting to each action image the corresponding neuter image of the subject.
        In order to augment the pair considered we interpolate the sample between the neuter images and the action images
        with random rates (actually three random rates and the rate `1`) . Since a neural network need a lot of data to work properly we perform
        a data augmentation flipping, rotating and cropping the images.
        In order to work this function need the dataset image files in format: `name_action.ext`

        We use this generator for dynamically create the training dataset so that the neural network didn't see the same
        image twice.

        :param src_path: input folder
        :param batch_size: number of samples for each batch
        :param normalize_eyes: if True normalize landmarks with respect to eyes and the nose instead of the whole landmarks' bounding box
        :return:
            * batch_landmarks_matrix: matrix which contains all the normalized landmarks in batch, dim: [batch_size, 68 * 2]
            * training_pairs_labels: matrix with the labels for each action in batch, dim: [batch_size, num_actions]
      """

    MIN_ROTATION_DEGREE = -14
    MAX_ROTATION_DEGREE = 14

    CROPPING_TOLERANCE = 0.7

    neuter_landmarks = {}

    subjects = sorted(set(map(lambda i: parse_image_path(i)[0], src_path.iterdir())))
    actions = sorted(set(map(lambda i: parse_image_path(i)[1], src_path.iterdir())))

    face_to_label_matrix = np.identity(len(actions))
    face_row_mapping = {}
    for i, action in enumerate(sorted(list(actions))):
        face_row_mapping[action] = i

    for subject in subjects:
        print(f"Importing subject {subject}")
        subject_neuter_image_path = list(src_path.glob(f"{subject}_neutro.*"))[0]
        subject_neuter_image = cv2.imread(str(subject_neuter_image_path), cv2.IMREAD_UNCHANGED)
        landmarks_subject_image = extract_landmarks(subject_neuter_image)

        if not landmarks_subject_image:
            print(f"Neuter landmarks not found for subject {subject}. Skipping...", file=sys.stderr)
            subjects.remove(subject)

            continue

        if normalize_eyes:
            neuter_subject_landmarks = normalize_landmarks_eyes(landmarks_subject_image[0]).flatten()
        else:
            neuter_subject_landmarks = normalize_landmarks(landmarks_subject_image[0]).flatten()

        print(f"Extracted landmarks from {subject_neuter_image_path}")

        neuter_landmarks[subject] = neuter_subject_landmarks

        # flipping the image

        flipped_image = horizontal_flip(subject_neuter_image)
        landmarks_subject_flipped_image = extract_landmarks(flipped_image)
        if not landmarks_subject_flipped_image:
            print(f"Neuter landmarks not found for subject {subject}-flipped. Skipping...", file=sys.stderr)
            subjects.remove(subject)
            continue

        if normalize_eyes:
            neuter_subject_flipped_landmarks = normalize_landmarks_eyes(landmarks_subject_flipped_image[0]).flatten()
        else:
            neuter_subject_flipped_landmarks = normalize_landmarks(landmarks_subject_flipped_image[0]).flatten()

        print(f"Extracted landmarks flipped from {subject_neuter_image_path}")

        neuter_landmarks[f"{subject}_flipped"] = neuter_subject_flipped_landmarks

    array_subjects = np.array(list(subjects))
    array_actions = np.array(list(actions))
    flip_image = np.array([True, False])
    while True:
        batch_landmarks_matrix = np.empty((0, 68 * 2))
        batch_labels = np.empty((0, 8), dtype=np.int8)
        while len(batch_labels) < batch_size:
            current_action = np.random.choice(array_actions)
            current_subject = np.random.choice(array_subjects)
            do_flip_image = np.random.choice(flip_image)
            current_image_angle_rotation = random.randint(MIN_ROTATION_DEGREE, MAX_ROTATION_DEGREE)
            rates = np.random.rand(3)
            rates = np.append(rates, 1)

            image_name = f"{current_subject}_{current_action}"
            rough_image_path_list = list(src_path.glob(f"{image_name}.*"))
            if not rough_image_path_list:
               # print(f"file with name {image_name} not found", file=sys.stderr)
                continue

            rough_image_path = rough_image_path_list[0]
            image = cv2.imread(str(rough_image_path), cv2.IMREAD_UNCHANGED)
            if do_flip_image:
                image = horizontal_flip(image)
                current_neuter_landmark = neuter_landmarks.get(f"{current_subject}_flipped")
            else:
                current_neuter_landmark = neuter_landmarks.get(current_subject)

            if current_neuter_landmark is None:
                print(f"neuter image of subject {current_subject} not available", file=sys.stderr)
                continue

            image_landmarks = extract_landmarks(image)
            if not image_landmarks:
                print(f"No landmark found in {image_name} with flipping = {do_flip_image}", file=sys.stderr)
                continue

            # image cropping
            landmark_min_x = np.min(image_landmarks[0][:, 0])
            landmark_max_x = np.max(image_landmarks[0][:, 0])
            landmark_min_y = np.min(image_landmarks[0][:, 1])
            landmark_max_y = np.max(image_landmarks[0][:, 1])

            cropping_min_x_coordinate = random.randint(0, int(CROPPING_TOLERANCE * landmark_min_x))
            cropping_max_x_coordinate = random.randint(
                image.shape[1] - int(CROPPING_TOLERANCE * (image.shape[1] - landmark_max_x)),
                image.shape[1])
            cropping_min_y_coordinate = random.randint(0, int(CROPPING_TOLERANCE * landmark_min_y))
            cropping_max_y_coordinate = random.randint(
                image.shape[0] - int(CROPPING_TOLERANCE * (image.shape[0] - landmark_max_y)),
                image.shape[0])

            image = image[cropping_min_y_coordinate:cropping_max_y_coordinate,
                    cropping_min_x_coordinate: cropping_max_x_coordinate, :]

            # rotating image
            final_image = rotate_image(image, current_image_angle_rotation)

            final_image_landmarks = extract_landmarks(final_image)
            if not final_image_landmarks:
                print(
                    f"No landmark found in {image_name} rotate of {current_image_angle_rotation} cropped and with flip = {do_flip_image}",
                    file=sys.stderr)
                continue

            if normalize_eyes:
                normalized_landmarks = normalize_landmarks_eyes(final_image_landmarks[0]).flatten()
            else:
                normalized_landmarks = normalize_landmarks(final_image_landmarks[0]).flatten()

            for rate in rates:
                rescaled_interpolated_landmark = interpolate_landmarks(current_neuter_landmark,
                                                                       normalized_landmarks,
                                                                       rate) - current_neuter_landmark
                batch_landmarks_matrix = np.vstack([batch_landmarks_matrix, rescaled_interpolated_landmark])
                if "sx" == current_action[-2:] and do_flip_image:
                    current_action = current_action[:-2] + "dx"
                elif "dx" == current_action[-2:] and do_flip_image:
                    current_action = current_action[:-2] + "sx"

                batch_labels = np.vstack([batch_labels, rate * face_to_label_matrix[face_row_mapping[current_action]] +
                                          (1 - rate) * face_to_label_matrix[face_row_mapping["neutro"]]])

        yield batch_landmarks_matrix, batch_labels


callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=25,
        mode='auto',
        #   min_delta=-0.0001
    ),
    keras.callbacks.ModelCheckpoint(
        filepath='./neural_model/saved-models-{epoch:06d}-{val_loss:.5f}.h5',
        monitor='val_loss',
        save_best_only=False
    ),
    keras.callbacks.CSVLogger(
        filename='./neural_model/my_model.csv',
        separator=',',
        append=True
    ),
]


def main():
    batch_size = 200
    neural_model_path = Path("neural_model")
    neural_model_path.mkdir(exist_ok=True)

    neural_validation_landmarks_path = Path("neural_validation_landmarks.npy")
    neural_validation_labels_path = Path("neural_validation_labels.npy")

    if not neural_validation_labels_path.exists() or not neural_validation_landmarks_path.exists():
        print("Generating validation dataset...")

        neural_validation_landmarks, neural_validation_labels = generate_neural_network_dataset(
            Path("dataset_neural_validation").resolve(), [0.2, 0.4, 0.6, 0.8, 1])
        np.save(str(neural_validation_landmarks_path), neural_validation_landmarks)
        np.save(str(neural_validation_labels_path), neural_validation_labels)
    else:
        neural_validation_landmarks = np.load(str(neural_validation_landmarks_path))
        neural_validation_labels = np.load(str(neural_validation_labels_path))

    training_generator = dataset_generator(Path("dataset_neural_training").resolve(), batch_size=batch_size)
    validation_data = tuple([neural_validation_landmarks, neural_validation_labels])

    model = keras.Sequential()
    model.add(keras.layers.Input(batch_input_shape=[None, 68 * 2]))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(64, kernel_initializer="he_normal", activation="elu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(64, kernel_initializer="he_normal", activation="elu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(32, kernel_initializer="he_normal", activation="elu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(8, activation="sigmoid"))  # 8 = num_actions, change if you use more actions

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit_generator(generator=training_generator,
                                  steps_per_epoch=50,
                                  epochs=100,
                                  verbose=1,
                                  validation_data=validation_data,
                                  callbacks=callbacks_list)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./neural_model/training-validation-loss')

    plt.clf()
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.savefig('./neural_model/training-validation-accuracy')


if __name__ == '__main__':
    main()
