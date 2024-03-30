# BrailleNet: Custom Light Weight Convolutional Neural Network (CNN) for Braille Character Classification

## Introduction

BrailleNet is a convolutional neural network (CNN) designed for the classification of Braille characters. This repository includes the code for training the model using the Braille Dataset and provides an overview of the network architecture.

## Dataset Preparation

1. The Braille Dataset is organized in the following structure:

    ```
    dataset/
        └── Braille Dataset/
            └── Braille Dataset/
                ├── a1.JPG
                ├── a2.JPG
                ├── ...
                └── z9.JPG
    ```

## Model Architecture

The CNN architecture includes separable convolutional layers, batch normalization, and dropout for robust feature extraction. The model is trained on Braille characters with data augmentation.

![BrailleNet Architecture](Model%20Architecture/BrailleNet.png)

## Code

Here is a Python file called 'brailleNet_CNN.py' is used to train the model for Braille Character Classification.

## Evaluation

After training, the model is evaluated on a validation set. Below is a summary of the model evaluation:

| Metric    | Value   |
|-----------|---------|
| Accuracy  | 97.44%  |
| Recall    | 96.78%  |
| F1-Score  | 96.81%  |
| Precision | 95.67%  |

## Requirements

- Python 3.x
- TensorFlow
- Keras
- Matplotlib (for visualization)

Feel free to customize the model architecture or experiment with hyperparameters based on specific requirements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
