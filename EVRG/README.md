# Image-to-Report Generation Project

This project aims to develop a system that generates textual reports (captions) for images. It leverages a two-stage approach:

1.  **Hugging Face BLIP Model for Initial Captioning**: Utilizing a pre-trained BLIP model to generate initial captions for a large dataset of images. This step is crucial for bootstrapping a custom image-report dataset.

2.  **Custom Image Analysis and Report Generation Models**: Training two interconnected custom models on the generated dataset:
    *   **`ModelVision`**: An image analysis model responsible for extracting visual features from images.
    *   **`ModelText`**: A report generation model that takes the visual features and generates a descriptive caption.

## Project Structure and Workflow

1.  **Environment Setup**: Installs necessary libraries (`transformers`, `torchtext`, `kagglehub`).
2.  **Dataset Download**: Downloads the MSCOCO dataset using `kagglehub`.
3.  **Image Loading**: Loads and lists all `.jpg` image files from the downloaded dataset.
4.  **Initial Caption Generation**: Uses the Hugging Face `BlipForConditionalGeneration` model to create a preliminary report (caption) for each image. These captions form the basis of our custom training data.
5.  **Data Preparation**: Tokenizes the generated captions using the BLIP tokenizer and creates a custom `CombinedImageTextDataset` for training the `ModelVision` and `ModelText` models.
6.  **Model Definition**: Defines the `ModelVision` (CNN-based image encoder) and `ModelText` (Transformer-like decoder with cross-attention) architectures.
7.  **Model Training**: Trains the `ModelVision` and `ModelText` collaboratively using an Adam optimizer and CrossEntropyLoss.
8.  **Inference**: Provides a function (`EVRG`) to take a new image and generate a report using the trained `ModelVision` and `ModelText` models.

## Key Components

*   **`BlipProcessor`, `BlipForConditionalGeneration`**: From Hugging Face, used for initial caption generation.
*   **`ImageDataset`**: A custom PyTorch `Dataset` for loading images.
*   **`CombinedImageTextDataset`**: A custom PyTorch `Dataset` that pairs images with their tokenized captions for training.
*   **`ModelVision`**: A custom Convolutional Neural Network (CNN) for image feature extraction.
*   **`ModelText`**: A custom model employing embedding layers, self-attention, and cross-attention mechanisms to generate text reports based on image features.
*   **`DataLoader`**: Used for efficient batching of data during training.

## Getting Started

To run this project:

1.  **Execute cells sequentially** from top to bottom.
2.  **Monitor Training**: Training can be time-consuming, especially with a large dataset. The training loop ` is initially configured for 5 epochs for quicker testing. You can adjust the `EPOCHS` variable for longer training, or uncomment the `imgs = imgs[:5000]` line  to train on a smaller subset of images for faster iteration.
3.  **Upload Image for Inference**: After training, you can upload your own image in the final cell  to get a generated report.

**Note**: Training on the full MSCOCO dataset for many epochs can be extremely time-consuming, especially without GPU acceleration. The current setup is configured for faster iteration and testing with reduced epochs.
