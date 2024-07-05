# MNIST Digit Recognition with a Deep Neural Network (From Scratch)

## Description

This project implements a three-layer deep neural network (DNN) from scratch using Python and NumPy to classify handwritten digits from the MNIST dataset. The goal is to build a model that can accurately predict the digit (0-9) represented in an image. This project was completed as part of my journey to learn the fundamentals of artificial neural networks (ANNs).

## Features

- Implementation of a DNN with one hidden layer.
- Forward and backward propagation algorithms for training.
- ReLU activation function for the hidden layer.
- Softmax activation function for the output layer.
- Gradient descent for optimization.
- Evaluation metrics including accuracy on both training and test sets. 
- Visualization of a randomly chosen image prediction.
- Cost function plot to monitor training progress.

## How to Run the Code

1. **Prerequisites:** Make sure you have Python installed along with the following libraries:
   - NumPy
   - Matplotlib

2. **Clone the repository:**
   ```bash
   git clone https://github.com/ratulbanik/mnist-dnn-from-scratch.git

3. **Navigate to the project directory:**
   ```bash
   cd mnist-dnn-from-scratch

4. **Run the Jupyter Notebook:**
   ```bash
   jupyter notebook DNN_Final.ipynb

## Results

The model achieved the following accuracy:
- **Training Accuracy:** 100.00%
- **Test Accuracy:** 88.29%

## Future Work

- Experiment with different hyperparameters (learning rate, hidden layer size, etc.) to potentially improve performance.
- Explore other activation functions (e.g., tanh, sigmoid).
- Implement more advanced optimization algorithms (e.g., Adam, RMSprop).
- Add additional evaluation metrics (precision, recall, F1-score, confusion matrix).
- Improve the efficiency of the code.

## Acknowledgments

- The code and mathematical calcualtions are inspired from **Coding Lane** YouTube channel (https://youtube.com/playlist?list=PLuhqtP7jdD8CftMk831qdE8BlIteSaNzD&si=svZJJDjoJUcCPKpG).
- The explanation of the softmax derivative was based on the article (https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/).
