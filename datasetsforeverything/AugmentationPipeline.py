import shutil
import Augmentor
import os
import shutil

def augmentation_pipeline(DIRECTORY, NUMBER_TO_SAMPLE):
    p = Augmentor.Pipeline(DIRECTORY)
    p.flip_left_right(probability=0.5)
    p.rotate(
        probability=0.5,
        max_left_rotation=15,
        max_right_rotation=15
    )
    p.zoom(
        probability=0.5,
        min_factor=1.05,
        max_factor=1.25
    )

    p.sample(NUMBER_TO_SAMPLE)

    for file in os.listdir(os.path.join(DIRECTORY, 'output')):
        shutil.move(os.path.join(DIRECTORY, 'output', file), DIRECTORY)
    