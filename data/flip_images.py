from preprocessing import flip_images
import os
import shutil

source_dir = "/media/adam/T9/ood_slam_data/datasets/kitti/odometry_gray/sequences"
flipped_dir = "/media/adam/T9/ood_slam_data/datasets/kitti/odometry_gray/flipped_data/sequences"

if os.path.exists(flipped_dir):
    shutil.rmtree(flipped_dir)

flip_images(source_dir, flipped_dir)


print("Flipping completed successfully!")