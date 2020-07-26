import sys
from pathlib import Path
from typing import List

import cv2
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import random

from utils import parse_image_path, extract_landmarks, normalize_landmarks, interpolate_landmarks, \
    normalize_landmarks_eyes

import skimage as sk
from skimage import transform
from skimage import util


def rotate_image(image_array: np.ndarray, degree: int):
    return sk.transform.rotate(image_array, degree, preserve_range=True).astype(np.uint8)


def horizontal_flip(image_array):
    return image_array[:, ::-1]


def generate_neural_network_dataset(src_path: Path, rates: List[int]):
    face_to_label_matrix = np.identity(8)
    face_row_mapping = {}
    face_row_mapping["neutro"] = 0
    face_row_mapping["occhiolinodx"] = 1
    face_row_mapping["occhiolinosx"] = 2
    face_row_mapping["cruccio"] = 3
    face_row_mapping["gengive"] = 4
    face_row_mapping["bacio"] = 5
    face_row_mapping["sorriso"] = 6
    face_row_mapping["sorrisino"] = 7

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
    actions.remove("neutro")

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

        neuter_subject_landmarks = normalize_landmarks_eyes(landmarks_subject_image[0]).flatten()
        print(f"Extracted landmarks from {subject_neuter_image_path}")

        neuter_landmarks[subject] = neuter_subject_landmarks
        # landmarks_matrix = np.vstack([landmarks_matrix, neuter_subject_landmarks])
        for ith_cropping in range(5):
            cropping_min_x_coordinate = random.randint(0, int(0.5 * landmark_min_x))
            cropping_max_x_coordinate = random.randint(int(0.5 * (subject_neuter_image.shape[1] + landmark_max_x)),
                                                       subject_neuter_image.shape[1])
            cropping_min_y_coordinate = random.randint(0, int(0.5 * landmark_min_y))
            cropping_max_y_coordinate = random.randint(int(0.5 * (subject_neuter_image.shape[0] + landmark_max_y)),
                                                       subject_neuter_image.shape[0])

            cropped_image = subject_neuter_image[cropping_min_y_coordinate:cropping_max_y_coordinate,
                            cropping_min_x_coordinate: cropping_max_x_coordinate, :]

            for degree in range(-10, 12, 2):
                rotated_image = rotate_image(cropped_image, degree)
                landmarks_cropped_rotated_found = extract_landmarks(rotated_image)
                if not landmarks_cropped_rotated_found:
                    print(
                        f"Neuter landmarks not found for subject {subject}. At angle {degree} at {ith_cropping} Skipping...",
                        file=sys.stderr)
                    action_images = list(filter(lambda i: subject not in i.name, list(action_images)))

                else:
                    normalized_landmarks = normalize_landmarks_eyes(landmarks_cropped_rotated_found[0]).flatten()
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

        for ith_cropping in range(5):
            cropping_min_x_coordinate = random.randint(0, int(0.5 * landmark_min_x))
            cropping_max_x_coordinate = random.randint(int(0.5 * (subject_action_image.shape[1] + landmark_max_x)),
                                                       subject_action_image.shape[1])
            cropping_min_y_coordinate = random.randint(0, int(0.5 * landmark_min_y))
            cropping_max_y_coordinate = random.randint(int(0.5 * (subject_action_image.shape[0] + landmark_max_y)),
                                                       subject_action_image.shape[0])

            cropped_image = subject_action_image[cropping_min_y_coordinate:cropping_max_y_coordinate,
                            cropping_min_x_coordinate: cropping_max_x_coordinate, :]

            for degree in range(-10, 12, 2):
                rotated_image = rotate_image(cropped_image, degree)
                landmarks_cropped_rotated_found = extract_landmarks(rotated_image)
                if not landmarks_cropped_rotated_found:
                    print(
                        f"Action landmarks ({action}) not found for subject {subject}. At angle {degree} at {ith_cropping} Skipping...",
                        file=sys.stderr)
                else:
                    action_subject_landmarks = normalize_landmarks_eyes(landmarks_cropped_rotated_found[0]).flatten()
                    neuter_subject_landmarks = neuter_landmarks[subject]

                    for rate in rates:
                        interpolated_subject_landmarks = interpolate_landmarks(neuter_subject_landmarks,
                                                                               action_subject_landmarks,
                                                                               rate) - neuter_subject_landmarks
                        landmarks_matrix = np.vstack([landmarks_matrix, interpolated_subject_landmarks])
                        labels = np.vstack([labels, rate * face_to_label_matrix[face_row_mapping[action]] +
                                            (1 - rate) * face_to_label_matrix[face_row_mapping["neutro"]]])

    return landmarks_matrix, labels


def dataset_generator(src_path: Path, rates: List[int], batch_size: int):
    face_to_label_matrix = np.identity(8)
    face_row_mapping = {}
    face_row_mapping["neutro"] = 0
    face_row_mapping["occhiolinodx"] = 1
    face_row_mapping["occhiolinosx"] = 2
    face_row_mapping["cruccio"] = 3
    face_row_mapping["gengive"] = 4
    face_row_mapping["bacio"] = 5
    face_row_mapping["sorriso"] = 6
    face_row_mapping["sorrisino"] = 7

    if 0 in rates:
        rates.remove(0)
    if 1 not in rates:
        rates.append(1)

    neuter_landmarks = {}

    subjects = sorted(set(map(lambda i: parse_image_path(i)[0], src_path.iterdir())))
    actions = sorted(set(map(lambda i: parse_image_path(i)[1], src_path.iterdir())))
    actions.remove("neutro")

    for subject in subjects:
        print(f"Importing subject {subject}")
        subject_neuter_image_path = list(src_path.glob(f"{subject}_neutro.*"))[0]
        subject_neuter_image = cv2.imread(str(subject_neuter_image_path), cv2.IMREAD_UNCHANGED)
        landmarks_subject_image = extract_landmarks(subject_neuter_image)

        if not landmarks_subject_image:
            print(f"Neuter landmarks not found for subject {subject}. Skipping...", file=sys.stderr)
            # action_images = list(filter(lambda i: subject not in i.name, list(action_images)))
            subjects.remove(subject)

            continue

        neuter_subject_landmarks = normalize_landmarks_eyes(landmarks_subject_image[0]).flatten()
        print(f"Extracted landmarks from {subject_neuter_image_path}")

        neuter_landmarks[subject] = neuter_subject_landmarks

        # flipping the image

        flipped_image = horizontal_flip(subject_neuter_image)
        landmarks_subject_flipped_image = extract_landmarks(flipped_image)
        if not landmarks_subject_flipped_image:
            print(f"Neuter landmarks not found for subject {subject}-flipped. Skipping...", file=sys.stderr)
            # action_images = list(filter(lambda i: subject not in i.name, list(action_images)))
            subjects.remove(subject)
            continue

        neuter_subject_flipped_landmarks = normalize_landmarks_eyes(landmarks_subject_flipped_image[0]).flatten()
        print(f"Extracted landmarks flipped from {subject_neuter_image_path}")

        neuter_landmarks[f"{subject}_flipped"] = neuter_subject_flipped_landmarks

    array_subjects = np.array(list(subjects))
    array_actions = np.array(list(actions))
    array_actions = np.append(array_actions, ["neutro"])
    flip_image = np.array([True, False])
    while True:
        batch_landmarks_matrix = np.empty((0, 68 * 2))
        batch_labels = np.empty((0, 8), dtype=np.int8)
        while len(batch_labels) < batch_size:
            current_action = np.random.choice(array_actions)
            current_subject = np.random.choice(array_subjects)
            do_flip_image = np.random.choice(flip_image)
            current_image_angle_rotation = random.randint(-12, 12)

            # if "occhiolino" in current_action:
            #     do_flip_image = False

            image_name = f"{current_subject}_{current_action}"
            rough_image_path_list = list(src_path.glob(f"{image_name}.*"))
            if not rough_image_path_list:
                print(f"file with name {image_name} not found", file=sys.stderr)
                continue

            rough_image_path = rough_image_path_list[0]
            image = cv2.imread(str(rough_image_path), cv2.IMREAD_UNCHANGED)
            if do_flip_image:
                image = horizontal_flip(image)
                current_neuter_landmark = neuter_landmarks[f"{current_subject}_flipped"]
            else:
                current_neuter_landmark = neuter_landmarks[current_subject]

            image_landmarks = extract_landmarks(image)
            if not image_landmarks:
                print(f"No landmark found in {image_name} with flipping = {do_flip_image}", file=sys.stderr)
                continue

            # image cropping
            landmark_min_x = np.min(image_landmarks[0][:, 0])
            landmark_max_x = np.max(image_landmarks[0][:, 0])
            landmark_min_y = np.min(image_landmarks[0][:, 1])
            landmark_max_y = np.max(image_landmarks[0][:, 1])

            cropping_min_x_coordinate = random.randint(0, int(0.5 * landmark_min_x))
            cropping_max_x_coordinate = random.randint(int(0.5 * (image.shape[1] + landmark_max_x)), image.shape[1])
            cropping_min_y_coordinate = random.randint(0, int(0.5 * landmark_min_y))
            cropping_max_y_coordinate = random.randint(int(0.5 * (image.shape[0] + landmark_max_y)), image.shape[0])

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

            normalized_landmarks = normalize_landmarks_eyes(final_image_landmarks[0]).flatten()

            for rate in rates:
                rescaled_interpolated_landmark = interpolate_landmarks(current_neuter_landmark,
                                                                       normalized_landmarks,
                                                                       rate) - current_neuter_landmark
                batch_landmarks_matrix = np.vstack([batch_landmarks_matrix, rescaled_interpolated_landmark])
                if "occhiolino" in current_action and do_flip_image:
                    if "occhiolinodx" in current_action:
                        current_action = "occhiolinosx"
                    else:
                        current_action = "occhiolinodx"

                batch_labels = np.vstack([batch_labels, rate * face_to_label_matrix[face_row_mapping[current_action]] +
                                          (1 - rate) * face_to_label_matrix[face_row_mapping["neutro"]]])

        yield batch_landmarks_matrix, batch_labels


callbacks_list = [
    # keras.callbacks.EarlyStopping(
    #     monitor='loss',
    #     patience=1000000,
    #     mode='min',
    #     # min_delta=-0.0001
    # ),
    keras.callbacks.ModelCheckpoint(
        filepath='./Model/saved-models-{epoch:06d}-{val_loss:.5f}.h5',
        monitor='val_loss',
        save_best_only=False
    ),
    keras.callbacks.CSVLogger(
        filename='./Model/my_model.csv',
        separator=',',
        append=True
    ),
]


def main():
    # saved_landmarks_matrix = Path("landmarks_matrix_neural.npy")
    # saved_labels = Path("labels_neural.npy")
    #
    # if not saved_landmarks_matrix.exists() or not saved_labels.exists():
    #     print("Generating dataset...")
    #
    #     landmark_matrix, labels = generate_neural_network_dataset(Path("dataset_filtered").resolve(), [1])
    #
    #     np.save(str(saved_landmarks_matrix), landmark_matrix)
    #     np.save(str(saved_labels), labels)
    # else:
    #     print("Loading dataset from filesystem...")
    #     landmark_matrix = np.load(saved_landmarks_matrix)
    #     labels = np.load(saved_labels)
    #
    # print("Starting training...")

    batch_size = 50

    training_generator = dataset_generator(Path("dataset_neural_training").resolve(), [1], batch_size=batch_size)
    validation_data = generate_neural_network_dataset(Path("dataset_neural_validation").resolve(), [1])
    np.save("neural_validation_landmark_[1].npy", validation_data[0])
    np.save("neural_validation_labels_[1].npy", validation_data[1])

    model = keras.Sequential()
    model.add(keras.layers.Input(batch_input_shape=[None, 68 * 2]))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(64, kernel_initializer="he_normal", activation="elu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(32, kernel_initializer="he_normal", activation="elu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(16, kernel_initializer="he_normal", activation="elu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(8, activation="sigmoid"))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # history = model.fit(landmark_matrix, labels, epochs=200, callbacks=callbacks_list, verbose=1, shuffle=True,
    # validation_split = 0.15, batch_size = 25)

    history = model.fit_generator(generator=training_generator,
                                  steps_per_epoch=50,
                                  epochs=200,
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
    plt.savefig('./Model/training-validation-loss')

    plt.clf()
    mAP = history.history['accuracy']
    val_mAP = history.history['val_accuracy']
    plt.plot(epochs, mAP, 'bo', label='Training f1m')
    plt.plot(epochs, val_mAP, 'b', label='Validation f1m')
    plt.title('Training and validation f1m  ')
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.savefig('./Model/training-validation-f1m')


if __name__ == '__main__':
    main()
