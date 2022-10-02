import pickle

import numpy as np
from fern_cascade import FernCascade
from utils import get_mean_shape, project_shape, reproject_shape


class ShapeRegressor:
    def __init__(self):
        self._bounding_box = None
        self._training_shapes = None
        self._first_level_num = None
        self._landmark_num = None
        self._fern_cascades = None
        self._mean_shape = None

    def train(
        self,
        images,
        ground_truth_shapes,
        bounding_box,
        first_level_num,
        second_level_num,
        candidate_pixel_num,
        fern_pixel_num,
        initial_number,
    ):
        print("Start training...")
        self._bounding_box = bounding_box
        self._training_shapes = ground_truth_shapes.copy()
        self._first_level_num = first_level_num
        self._landmark_num = ground_truth_shapes[0].shape[0]

        augmented_images = []
        augmented_bounding_box = []
        augmented_ground_truth_shapes = []
        current_shapes = []

        for i in range(images.shape[0]):
            for j in range(initial_number):
                index = 0
                while index == i:
                    index = np.random.randint(0, images.shape[0])
                augmented_images.append(images[i])
                augmented_ground_truth_shapes.append(ground_truth_shapes[i])
                augmented_bounding_box.append(bounding_box[i])

                temp = ground_truth_shapes[index]
                temp = project_shape(temp, bounding_box[index])
                temp = reproject_shape(temp, bounding_box[i])
                current_shapes.append(temp)

        self._mean_shape = get_mean_shape(ground_truth_shapes, bounding_box)
        self._fern_cascades = []

        for i in range(first_level_num):
            print(f"Training fern cascades: {i + 1} out of {first_level_num}")
            self._fern_cascades.append(FernCascade())
            prediction = self._fern_cascades[i].train(
                np.array(augmented_images),
                np.array(current_shapes),
                augmented_ground_truth_shapes,
                augmented_bounding_box,
                self._mean_shape,
                second_level_num,
                candidate_pixel_num,
                fern_pixel_num,
                i + 1,
                first_level_num,
            )
            for j in range(prediction.shape[0]):
                current_shapes[j] = prediction[j] + project_shape(
                    current_shapes[j], augmented_bounding_box[j]
                )
                current_shapes[j] = reproject_shape(
                    current_shapes[j], augmented_bounding_box[j]
                )

    def predict(self, image, bounding_box, initial_num):
        result = np.zeros((self._landmark_num, 2))

        for i in range(initial_num):
            index = int(np.random.uniform(0, len(self._training_shapes)))
            current_shape = self._training_shapes[index]
            current_bounding_box = self._bounding_box[index]

            current_shape = project_shape(current_shape, current_bounding_box)
            current_shape = reproject_shape(current_shape, bounding_box)

            for fern_cascade in self._fern_cascades:
                prediction = fern_cascade.predict(
                    image, bounding_box, self._mean_shape, current_shape
                )

                current_shape = prediction + project_shape(current_shape, bounding_box)
                current_shape = reproject_shape(current_shape, bounding_box)

            result += current_shape

        return 1.0 / initial_num * result

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            shape_regressor = pickle.load(f)
        return shape_regressor

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
