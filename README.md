<!DOCTYPE html>
<html lang="en">
<head>
    
</head>
<body>

<h1>BrailleNet: Custom Light Weight Convolutional Neural Network (CNN) for Braille Character Classification</h1>

<h2>Introduction</h2>
<p>BrailleNet is a convolutional neural network (CNN) designed for the classification of Braille characters. This repository includes the code for training the model using the Braille Dataset and provides an overview of the network architecture.</p>

<h2>Dataset Preparation</h2>
<ol>
    <li>The Braille Dataset is organized in the following structure:</li>
    <code>dataset/<br>
        └── Braille Dataset/<br>
        &emsp;└── Braille Dataset/<br>
        &emsp;&emsp;├── a1.JPG<br>
        &emsp;&emsp;├── a2.JPG<br>
        &emsp;&emsp;├── ...<br>
        &emsp;&emsp;└── z9.JPG<br></code>
</ol>

<h2>Model Architecture</h2>
<p>The CNN architecture includes separable convolutional layers, batch normalization, and dropout for robust feature extraction. The model is trained on Braille characters with data augmentation.</p>

<h2>Code</h2>
<p>Here is a Python file called 'brailleNet_CNN.py' is used to train the model for Braille Character Classification.</p>

<h2>Evaluation</h2>
<ol>
    <li>After training, the model is evaluated on a validation set, and the accuracy is printed.</li>
</ol>

<h2>Requirements</h2>
<ul>
    <li>Python 3.x</li>
    <li>TensorFlow</li>
    <li>Keras</li>
    <li>Matplotlib (for visualization)</li>
</ul>

<p>Feel free to customize the model architecture or experiment with hyperparameters based on specific requirements.</p>

<h2>License</h2>
<p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>

</body>
</html>
