import cv2
import numpy as np
from bounding_box import BoundingBox
from shape_regressor import ShapeRegressor

test_images = []
test_bounding_box = []
test_img_num = 507
initial_number = 20
landmark_num = 29

for i in range(test_img_num):
    image_name = "data/trainingImages/"
    image_name += f"{i + 1}.jpg"
    temp = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    test_images.append(temp)


with open("data/boundingbox_test.txt") as file:
    for i in range(test_img_num):
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
        test_bounding_box.append(temp)

model_path = "models/model.pkl"
regressor = ShapeRegressor.load(model_path)

while True:
    index = int(input("Image index: "))

    current_shape = regressor.predict(
        test_images[index], test_bounding_box[index], initial_number
    )

    image_copy = test_images[index].copy()

    for i in range(landmark_num):
        image_copy = cv2.circle(image_copy, current_shape[i], 3, (255, 0, 0), -1, 8, 0)

    cv2.imshow(image_copy)
    cv2.waitKey(0)
