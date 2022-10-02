import cv2
import numpy as np
from bounding_box import BoundingBox
from shape_regressor import ShapeRegressor

img_num = 1345
candidate_pixel_num = 400
fern_pixel_num = 5
first_level_num = 10
second_level_num = 500
landmark_num = 29
initial_number = 20
images = []

print("read images...")
for i in range(img_num):
    image_name = "data/trainingImages/"
    image_name += f"{i + 1}.jpg"
    temp = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    images.append(temp)

ground_truth_shapes = []
bounding_box = []

with open("data/boundingbox.txt") as file:
    for i in range(img_num):
        line = file.readline()
        start_x, start_y, width, height = [float(item) for item in line.split()]
        centroid_x = start_x + width / 2.0
        centroid_y = start_y + height / 2.0
        temp = BoundingBox(
            start_x=start_x,
            start_y=start_y,
            width=width,
            height=height,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
        )
        bounding_box.append(temp)

ground_truth_shapes = []
with open("data/keypoints.txt") as file:
    for i in range(img_num):
        line = file.readline()
        key_points = [float(item) for item in line.split()]
        temp = np.ndarray((landmark_num, 2))
        temp[:, 0] = key_points[:29]
        temp[:, 1] = key_points[29:]
        ground_truth_shapes.append(temp)
# OK
regressor = ShapeRegressor()
regressor.train(
    np.asarray(images),
    np.asarray(ground_truth_shapes),
    bounding_box,
    first_level_num,
    second_level_num,
    candidate_pixel_num,
    fern_pixel_num,
    initial_number,
)
regressor.save("models/model.pkl")
