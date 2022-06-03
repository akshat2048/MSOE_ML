"""

This Jupyter Notebook will recursively trace out the provided directory, scan for image files, and then invert them. This will NOT create a new copy of the images, so make sure you do that first.

---

Set DIRECTORY_WITH_IMAGES_TO_INVERT to the path where the images are located

For example:

    DIRECTORY_WITH_IMAGES_TO_INVERT = "/Users/akshatchannashetti/Downloads/test"
    - test
        - AP
        - L
        - PA
            - 1.png
            - 2.png
    

"""

DIRECTORY_WITH_IMAGES_TO_INVERT = "/Volumes/90OL67YGN/images"

import os
from PIL import Image, ImageEnhance
def calculate_brightness(image):

    greyscale_image = image.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):

        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)
    return 1 if brightness == 255 else brightness / scale

def brightness_factor_calculator(brightness_level, brightness_average):
    brightness_factor = brightness_average - brightness_level
    readjustment_factor = 0
    if brightness_factor < 0:
        readjustment_factor = 1 - brightness_factor
    elif brightness_factor > 0:
        readjustment_factor = 1 + brightness_factor
    else:
        readjustment_factor = 1
    return readjustment_factor
"""

Essentially you will calculate the average brightness of all the images in a directory, then you will adjust the brightness of all the images in that directory by that average.

"""

brightness = []
counter = 0

for root, dirs, fileBucket in os.walk(DIRECTORY_WITH_IMAGES_TO_INVERT):
    for file in fileBucket:
        if (file.endswith(".png") or file.endswith(".PNG") or file.endswith(".jpg") or file.endswith(".JPG") or file.endswith(".jpeg") or file.endswith(".JPEG")) and not file.startswith("."):
            print("Calculating brightness for image: " + file)
            image = Image.open(os.path.join(root, file))
            brightness.append(calculate_brightness(image))
            counter += 1

brightness_average = sum(brightness) / counter

counter = 0


for root, dirs, fileBucket in os.walk(DIRECTORY_WITH_IMAGES_TO_INVERT):
    for file in fileBucket:
        if (file.endswith(".png") or file.endswith(".PNG") or file.endswith(".jpg") or file.endswith(".JPG") or file.endswith(".jpeg") or file.endswith(".JPEG")) and not file.startswith("."):
            image = Image.open(os.path.join(root, file))
            brightness_i = brightness[counter]
            brightness_factor = brightness_factor_calculator(brightness_i, brightness_average)
            print("Adjusting brightness for image: " + file + " by factor: " + str(brightness_factor))
            output = ImageEnhance.Brightness(image).enhance(brightness_factor)
            os.remove(os.path.join(root, file))
            output.save(os.path.join(root, file))