import shutil
import Augmentor
import os
import shutil

def augmentation_pipeline(DIRECTORY, NUMBER_TO_SAMPLE):
    p = Augmentor.Pipeline(DIRECTORY)
    p.flip_left_right(probability=0.5)
    p.random_color(
        min_factor=0.9, 
        max_factor=0.99,
        probability=0.5
    )
    p.random_contrast(
        min_factor=0.9, 
        max_factor=0.99,
        probability=0.5
    )
    p.random_brightness(
        min_factor=0.9, 
        max_factor=0.99,
        probability=0.5
    )

    p.sample(NUMBER_TO_SAMPLE)

    for file in os.listdir(os.path.join(DIRECTORY, 'output')):
        shutil.move(os.path.join(DIRECTORY, 'output', file), DIRECTORY)
    