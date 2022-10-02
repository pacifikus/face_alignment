import cv2
import numpy as np


def get_mean_shape(shapes, bounding_box):
    result = np.zeros((shapes[0].shape[0], 2))
    for i in range(shapes.shape[0]):
        result += project_shape(shapes[i], bounding_box[i])
    result = 1.0 / shapes.shape[0] * result
    return result


def project_shape(shape, bounding_box):
    temp = np.ndarray((shape.shape[0], 2))
    for j in range(shape.shape[0]):
        temp[j, 0] = (shape[j, 0] - bounding_box.centroid_x) / (
            bounding_box.width / 2.0
        )
        temp[j, 1] = (shape[j, 1] - bounding_box.centroid_y) / (
            bounding_box.height / 2.0
        )
    return temp


def reproject_shape(shape, bounding_box):
    temp = np.ndarray((shape.shape[0], 2))
    for j in range(shape.shape[0]):
        temp[j, 0] = (shape[j, 0] - bounding_box.width) / (
            bounding_box.centroid_x / 2.0
        )
        temp[j, 1] = (shape[j, 1] - bounding_box.height) / (
            bounding_box.centroid_y / 2.0
        )
    return temp


def similarity_transform(shape1, shape2):
    rotation = np.zeros((2, 2))

    center_x_1, center_y_1, center_x_2, center_y_2 = 0, 0, 0, 0
    for i in range(shape1.shape[0]):
        center_x_1 += shape1[i, 0]
        center_y_1 += shape1[i, 1]
        center_x_2 += shape2[i, 0]
        center_x_2 += shape2[i, 1]

    center_x_1 /= shape1.shape[0]
    center_y_1 /= shape1.shape[0]
    center_x_2 /= shape2.shape[0]
    center_y_2 /= shape2.shape[0]

    temp1 = shape1.copy()
    temp2 = shape2.copy()

    for i in range(shape1.shape[0]):
        temp1[i, 0] -= center_x_1
        temp1[i, 1] -= center_y_1
        temp2[i, 0] -= center_x_2
        temp2[i, 1] -= center_y_2

    mean1 = np.ndarray(temp1.shape[0])
    mean2 = np.ndarray(temp2.shape[0])
    covariance1, mean1 = cv2.calcCovarMatrix(temp1.T, mean1, flags=cv2.COVAR_COLS)
    covariance2, mean2 = cv2.calcCovarMatrix(temp2.T, mean2, flags=cv2.COVAR_COLS)

    s1 = np.sqrt(np.linalg.norm(covariance1))
    s2 = np.sqrt(np.linalg.norm(covariance2))
    scale = s1 / s2
    temp1 = 1.0 / s1 * temp1
    temp2 = 1.0 / s2 * temp2
    num, den = 0, 0

    for i in range(shape1.shape[0]):
        num += temp1[i, 1] * temp2[i, 0] - temp1[i, 0] * temp2[i, 1]
        den += temp1[i, 0] * temp2[i, 0] - temp1[i, 1] * temp2[i, 1]

    norm = (num * num + den * den) ** 0.5
    sin_theta = num / norm
    cos_theta = den / norm
    rotation[0, 0] = cos_theta
    rotation[0, 1] = -sin_theta
    rotation[1, 0] = sin_theta
    rotation[1, 1] = cos_theta

    return rotation, scale


def calculate_covariance(v1, v2):

    v1 = v1 - np.mean(v1, dtype=np.float64)
    v2 = v2 - np.mean(v2, dtype=np.float64)

    return np.mean(np.dot(v1, v2))
