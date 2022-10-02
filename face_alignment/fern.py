import cv2
import numpy as np
from bounding_box import BoundingBox
from utils import calculate_covariance


class Fern:
    def __init__(self):
        self._fern_pixel_num = None
        self._landmark_num = None
        self._selected_nearest_landmark_index = None
        self._threshold = None
        self._selected_pixel_index = None
        self._selected_pixel_locations = None
        self._bin_output = None

    def train(
        self,
        candidate_pixel_intensity,
        covariance,
        candidate_pixel_locations,
        nearest_landmark_index,
        regression_targets: np.ndarray,
        fern_pixel_num: int,
    ):
        self._fern_pixel_num = fern_pixel_num
        self._landmark_num = regression_targets[0].shape[0]
        self._selected_pixel_index = np.zeros(shape=(fern_pixel_num, 2), dtype=int)
        self._selected_pixel_locations = np.zeros(
            shape=(fern_pixel_num, 4), dtype=float
        )
        self._selected_nearest_landmark_index = np.zeros(
            shape=(fern_pixel_num, 2), dtype=int
        )
        candidate_pixel_num = candidate_pixel_locations.shape[0]
        self._threshold = np.zeros(shape=(fern_pixel_num, 1), dtype=float)

        for i in range(self._fern_pixel_num):
            random_direction = np.random.uniform(-1.1, 1.1, (self._landmark_num, 2))

            random_direction = cv2.normalize(random_direction, random_direction)
            projection_result = np.zeros(regression_targets.shape[0])

            for j in range(regression_targets.shape[0]):
                projection_result[j] = (regression_targets[j] * random_direction).sum()

            covariance_projection_density = np.zeros((candidate_pixel_num, 1))
            for j in range(candidate_pixel_num):
                covariance_projection_density[j] = calculate_covariance(
                    projection_result, candidate_pixel_intensity[j]
                )

            max_correlation = -1
            max_pixel_index_1 = 0
            max_pixel_index_2 = 0
            for j in range(candidate_pixel_num):
                for k in range(candidate_pixel_num):
                    temp1 = covariance[j, j] + covariance[k, k] - 2 * covariance[j, k]
                    if abs(temp1) < 1e-10:
                        continue
                    flag = False
                    for p in range(i):
                        if (
                            j == self._selected_pixel_index[p][0]
                            and k == self._selected_pixel_index[p][1]
                        ):
                            flag = True
                            break
                        elif (
                            j == self._selected_pixel_index[p][1]
                            and k == self._selected_pixel_index[p][0]
                        ):
                            flag = True
                            break
                    if flag:
                        continue
                    temp = covariance_projection_density[
                        j
                    ] - covariance_projection_density[k] / (temp1**0.5)
                    if abs(temp) > max_correlation:
                        max_correlation = temp
                        max_pixel_index_1 = j
                        max_pixel_index_2 = k

            self._selected_pixel_index[i][0] = max_pixel_index_1
            self._selected_pixel_index[i][1] = max_pixel_index_2
            self._selected_pixel_locations[i][0] = candidate_pixel_locations[
                max_pixel_index_1
            ][0]
            self._selected_pixel_locations[i][1] = candidate_pixel_locations[
                max_pixel_index_1
            ][1]
            self._selected_pixel_locations[i][2] = candidate_pixel_locations[
                max_pixel_index_2
            ][0]
            self._selected_pixel_locations[i][3] = candidate_pixel_locations[
                max_pixel_index_2
            ][1]
            self._selected_nearest_landmark_index[i][0] = nearest_landmark_index[
                max_pixel_index_1
            ]
            self._selected_nearest_landmark_index[i][1] = nearest_landmark_index[
                max_pixel_index_2
            ]

            max_diff = -1
            for j in range(len(candidate_pixel_intensity[max_pixel_index_1])):
                temp = (
                    candidate_pixel_intensity[max_pixel_index_1][j]
                    - candidate_pixel_intensity[max_pixel_index_2][j]
                )
                if abs(temp) > max_diff:
                    max_diff = abs(temp)

            self._threshold[i] = np.random.uniform(-0.2 * max_diff, 0.2 * max_diff)

        bin_num = 2**fern_pixel_num

        shapes_in_bin = np.zeros((bin_num, regression_targets.shape[0]))
        for i in range(regression_targets.shape[0]):
            index = 0
            for j in range(fern_pixel_num):
                density_1 = candidate_pixel_intensity[self._selected_pixel_index[j][0]][
                    i
                ]
                density_2 = candidate_pixel_intensity[self._selected_pixel_index[j][1]][
                    i
                ]
                if density_1 - density_2 >= self._threshold[j]:
                    index += 2**j
            np.append(shapes_in_bin[index], i)

        prediction = np.ndarray(regression_targets.shape)
        self._bin_output = np.ndarray(
            (bin_num, regression_targets.shape[1], regression_targets.shape[2])
        )
        for i in range(bin_num):
            temp = np.zeros((self._landmark_num, 2))
            bin_size = shapes_in_bin[i].shape[0]
            for j in range(bin_size):
                index = shapes_in_bin[i][j]
                temp += regression_targets[int(index)]

            if bin_size == 0:
                self._bin_output[i] = temp
                continue

            temp = 1.0 / ((1.0 + 1000.0 / bin_size) * bin_size) * temp
            self._bin_output[i] = temp
            for j in range(bin_size):
                index = int(shapes_in_bin[i][j])
                prediction[index] = temp

        return prediction

    def predict(self, image, shape, rotation, bounding_box: BoundingBox, scale: float):
        index = 0
        for i in range(self._fern_pixel_num):
            nearest_landmark_index_1 = self._selected_nearest_landmark_index[i][0]
            nearest_landmark_index_2 = self._selected_nearest_landmark_index[i][1]
            x = self._selected_pixel_locations[i][0]
            y = self._selected_pixel_locations[i][1]
            project_x = (
                scale
                * (rotation[0][0] * x + rotation[0][1] * y)
                * bounding_box.width
                / 2.0
                + shape[nearest_landmark_index_1][0]
            )
            project_y = (
                scale
                * (rotation[1][0] * x + rotation[1][1] * y)
                * bounding_box.height
                / 2.0
                + shape[nearest_landmark_index_1][1]
            )

            project_x = max(0.0, min(project_x, image.shape[1] - 1.0))
            project_y = max(0.0, min(project_y, image.shape[0] - 1.0))
            intensity_1 = int(image[int(project_y), int(project_x)])

            x = self._selected_pixel_locations[i][2]
            y = self._selected_pixel_locations[i][3]
            project_x = (
                scale
                * (rotation[0][0] * x + rotation[0][1] * y)
                * bounding_box.width
                / 2.0
                + shape[nearest_landmark_index_2][0]
            )
            project_y = (
                scale
                * (rotation[1][0] * x + rotation[1][1] * y)
                * bounding_box.height
                / 2.0
                + shape[nearest_landmark_index_2][1]
            )
            project_x = max(0.0, min(project_x, image.shape[1] - 1.0))
            project_y = max(0.0, min(project_y, image.shape[0] - 1.0))
            intensity_2 = int(image[int(project_y), int(project_x)])

            if intensity_1 - intensity_2 >= self._threshold[i]:
                index += 2**i

        return self._bin_output[index]
