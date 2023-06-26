import gradio as gr
import numpy as np
import skimage.io as io
import skimage.metrics as metrics
import skimage.measure as measure
import skimage.transform as transform
from brisque import BRISQUE
import sewar
# Create an instance of BRISQUE
brisque_calculator = BRISQUE()

# Define the function to calculate image quality metrics
def calculate_image_quality(reference_image, distorted_image):
    # Resize images to have the same dimensions
    # reference_image = reference_image.astype(np.float32)
    # distorted_image = distorted_image.astype(np.float32)

    # distorted_image = transform.resize(distorted_image, reference_image.shape, anti_aliasing=True)
    # data_range = np.max([np.max(reference_image), np.max(distorted_image)])

    # Calculate Mean Squared Error (MSE)
    mse_score = sewar.mse(reference_image, distorted_image)

    # Calculate Root Mean Squared Error (RMSE)
    rmse_score = sewar.rmse(reference_image, distorted_image)

    # Calculate Structural Similarity Index (SSIM)
    ssim_score_sewar = sewar.ssim(reference_image, distorted_image)

    # Calculate PSNR
    psnr_score = metrics.peak_signal_noise_ratio(reference_image, distorted_image)


    # Calculate BRISQUE score
    brisque_score_reference = brisque_calculator.score(reference_image)
    brisque_score_distorted = brisque_calculator.score(distorted_image)

    return {
        'MSE': mse_score,
        'RMSE': rmse_score,
        'SSIM_SEWAR': ssim_score_sewar,
        'PSNR': psnr_score,
        'BRISQUE': {
            'reference': brisque_score_reference,
            'distorted': brisque_score_distorted,
        }
    }


def calculateNoReference(reference_image):
        # Calculate BRISQUE score
    brisque_score_reference = brisque_calculator.score(reference_image)

    return {
        'BRISQUE': brisque_score_reference
    }


layout1 = gr.Interface(
    fn=calculateNoReference,
    inputs=[ 
        gr.inputs.Image(label="Reference Image"),
        ],
    outputs='json',
    title= "No Reference"
)

# Create the layout for the second tab
layout2 = gr.Interface(
    fn=calculate_image_quality,
    inputs=[
        gr.inputs.Image(label="Reference Image"),
        gr.inputs.Image(label="Distorted Image")
    ],
    outputs='json',
    title="With Reference"
)

# Set up the tabs
layout = gr.TabbedInterface([layout1, layout2], ['no-reference', 'with-reference'])


# Launch the interface
layout.launch()
