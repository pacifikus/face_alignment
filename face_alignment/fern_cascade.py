import cv2
import numpy as np
from fern import Fern
from utils import project_shape, similarity_transform


class FernCascade:
    def __init__(self):
        self._ferns = None
        self._second_level_num = None

    def train(
        self,
        images,
        current_shapes,
        ground_truth_shapes,
        bounding_box,
        mean_shape,
        second_level_num,
        candidate_pixel_num,
        fern_pixel_num,
        curr_level_num,
        first_level_num,
    ):
        candidate_pixel_locations = np.zeros((candidate_pixel_num, 2))
        nearest_landmark_index = np.zeros((candidate_pixel_num, 1))
        self._second_level_num = second_level_num
        regression_targets = np.zeros(current_shapes.shape)
        for i in range(current_shapes.shape[0]):
            regression_targets[i] = project_shape(
                ground_truth_shapes[i], bounding_box[i]
            ) - project_shape(current_shapes[i], bounding_box[i])
            rotation, scale = similarity_transform(
                mean_shape, project_shape(current_shapes[i], bounding_box[i])
            )
            cv2.transpose(rotation, rotation)
            regression_targets[i] = scale * np.dot(regression_targets[i], rotation)

        for i in range(candidate_pixel_num):
            x = np.random.uniform(-1.0, 1.0)
            y = np.random.uniform(-1.0, 1.0)
            if x * x + y * y > 1.0:
                i -= 1
                continue
            min_dist = 1e10
            min_index = 0
            for j in range(mean_shape.shape[0]):
                temp = (mean_shape[j, 0] - x) ** 2 + (mean_shape[j, 1] - y) ** 2
                if temp < min_dist:
                    min_dist = temp
                    min_index = j

            candidate_pixel_locations[i, 0] = x - mean_shape[min_index, 0]
            candidate_pixel_locations[i, 1] = y - mean_shape[min_index, 1]
            nearest_landmark_index[i] = min_index

        densities = [[] for item in range(candidate_pixel_num)]
        for i in range(images.shape[0]):
            temp = project_shape(current_shapes[i], bounding_box[i])
            rotation, scale = similarity_transform(temp, mean_shape)
            for j in range(candidate_pixel_num):
                project_x = (
                    rotation[0, 0] * candidate_pixel_locations[j, 0]
                    + rotation[0, 1] * candidate_pixel_locations[j, 1]
                )
                project_y = (
                    rotation[1, 0] * candidate_pixel_locations[j, 0]
                    + rotation[1, 1] * candidate_pixel_locations[j, 1]
                )
                project_x = scale * project_x * bounding_box[i].width / 2.0
                project_y = scale * project_y * bounding_box[i].height / 2.0
                index = int(nearest_landmark_index[j])
                real_x = project_x + current_shapes[i][index, 0]
                real_y = project_y + current_shapes[i][index, 1]
                real_x = int(max(0.0, min(float(real_x), images[i].shape[1] - 1.0)))
                real_y = int(max(0.0, min(float(real_y), images[i].shape[0] - 1.0)))
                densities[j].append(int(images[i][real_y, real_x]))

        covariance = np.cov(densities)

        prediction = np.zeros((regression_targets.shape[0], mean_shape.shape[0], 2))
        # for i in range(regression_targets.shape[0]):
        #     prediction[i] = np.zeros((mean_shape.shape[0], 2))
        self._ferns = [Fern() for i in range(second_level_num)]

        for i in range(second_level_num):
            temp = self._ferns[i].train(
                densities,
                covariance,
                candidate_pixel_locations,
                nearest_landmark_index,
                regression_targets,
                fern_pixel_num,
            )
            for j in range(temp.shape[0]):
                prediction[j] += temp[j]
                regression_targets[j] -= temp[j]

            if i + 1 % 10 == 0:
                print(f"Fern cascades: {curr_level_num} out of {first_level_num};")
                print(f"Ferns: {i + 1} out of {second_level_num};")

        for i in range(prediction.shape[0]):
            rotation, scale = similarity_transform(
                project_shape(current_shapes[i], bounding_box[i]), mean_shape
            )
            cv2.transpose(rotation, rotation)
            prediction[i] = scale * np.dot(regression_targets[i], rotation)
        return prediction

    def predict(
        self,
        image,
        bounding_box,
        mean_shape,
        shape,
    ):
        result = np.zeros((shape.shape[0], 2))
        rotation, scale = similarity_transform(
            project_shape(shape, bounding_box), mean_shape
        )

        for fern in self._ferns:
            result += fern.predict(image, shape, rotation, bounding_box, scale)

        rotation, scale = similarity_transform(
            project_shape(shape, bounding_box), mean_shape
        )

        cv2.transpose(rotation, rotation)
        return scale * np.dot(result, rotation)
