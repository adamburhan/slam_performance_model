import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def load_sequence_images(base_path, seq):
    sequence_path = os.path.join(base_path, seq, 'image_0')

    if not os.path.exists(sequence_path):
        raise ValueError(f"Sequence folder {sequence_path} does not exist.")
    
    image_files = sorted([f for f in os.listdir(sequence_path) if f.endswith('.png')])

    images = []
    for img_file in image_files:
        img_path = os.path.join(sequence_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
        else:
            print(f"Warning: Unable to load image {img_file}")
    
    return images

def load_errors(base_path, seq):
    error_file = os.path.join(base_path, f'{seq}.txt')
    df = pd.read_csv(error_file, sep=' ', header=None, names=['frame', 'rotation_error', 'translation_error'])
    return df

def create_video_with_plots(images, errors_df, output_file, fps=5):
    height, width = images[0].shape
    
    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height + 300))  # Additional space for plots
    
    # Create a figure for plotting
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))  # Two subplots: bar and line plot

    # Loop over images and errors
    rotation_errors = []
    translation_errors = []

    for i, img in enumerate(images):
        frame = errors_df.iloc[i]['frame']
        rotation_error = errors_df.iloc[i]['rotation_error']
        translation_error = errors_df.iloc[i]['translation_error']
        
        # Append errors for dynamic plotting
        rotation_errors.append(rotation_error)
        translation_errors.append(translation_error)
        
        # Plot the bar graph (showing current frame's errors)
        ax[0].cla()  # Clear the previous bar plot
        ax[0].bar(['Rotation', 'Translation'], [rotation_error, translation_error], color=['blue', 'red'])
        ax[0].set_ylim(0, max(errors_df['rotation_error'].max(), errors_df['translation_error'].max()))  # Normalize the y-axis
        ax[0].set_title(f'Frame {int(frame)} Errors')
        
        # Plot the line graph (showing error trend over time)
        ax[1].cla()  # Clear the previous line plot
        ax[1].plot(rotation_errors, label='Rotation Error', color='blue')
        ax[1].plot(translation_errors, label='Translation Error', color='red')
        ax[1].set_xlim(0, len(images))  # Set x-axis limit to the number of frames
        ax[1].set_ylim(0, max(errors_df['rotation_error'].max(), errors_df['translation_error'].max()))  # Normalize y-axis
        ax[1].legend()
        ax[1].set_title('Error Trend Over Time')

        # Render the plot to an image
        fig.canvas.draw()
        plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Resize plot image to match the width of the video frame
        plot_image_resized = cv2.resize(plot_image, (width, 300))

        # Convert grayscale image to color to concatenate with plot
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Stack the video frame and the plot vertically
        combined_frame = np.vstack((img_color, plot_image_resized))

        # Write the combined frame to the video
        out.write(combined_frame)
    
    out.release()  # Finalize the video


base_path = '/media/adam/T9/ood_slam_data/datasets/kitti/odometry_gray/sequences'
base_path_errors = '/media/adam/T9/slam_performance_model/data/errors'
seq = '04'

# Load the images and the error data
images = load_sequence_images(base_path, seq)
errors_df = load_errors(base_path_errors, seq)

# Create the video with error plots
output_video_path = 'output_video_with_plots.mp4'
create_video_with_plots(images, errors_df, output_video_path, fps=5)

print(f"Video saved at {output_video_path}")
