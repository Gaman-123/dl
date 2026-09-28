# CS722L5C Deep Learning Lab Programs

This repository contains TensorFlow & Keras implementations of all 10 programs specified in the **CS722L5C Deep Learning Lab Manual** (Department of Computer Science & Engineering, SCEM, Mangaluru).

These files can be executed directly as standard Python scripts or copied cell-by-cell into a **Jupyter Notebook**.

---

## 📁 Repository Structure

| File | Program Description |
| :--- | :--- |
| [`01_xor_ffnn.py`](01_xor_ffnn.py) | **Exp 1:** Deep feedforward neural network to learn the XOR function using gradient-based learning and hidden units. |
| [`02_regularization_mnist.py`](02_regularization_mnist.py) | **Exp 2:** Regularization techniques (parameter norm penalties L1/L2, dataset augmentation, Gaussian noise, Dropout, EarlyStopping) on MNIST. |
| [`03_optimizers_comparison.py`](03_optimizers_comparison.py) | **Exp 3:** Comparison of optimization algorithms (SGD, Momentum, Adam) on a toy XOR dataset. |
| [`04_cnn_mnist.py`](04_cnn_mnist.py) | **Exp 4:** Convolutional Neural Network (CNN) on MNIST demonstrating convolution, pooling, and classification. |
| [`05_lstm_text_classification.py`](05_lstm_text_classification.py) | **Exp 5:** LSTM-based RNN for text classification on the IMDB sentiment dataset. |
| [`06_bidirectional_rnn.py`](06_bidirectional_rnn.py) | **Exp 6:** Bidirectional RNN vs Standard SimpleRNN comparison on sequence prediction tasks. |
| [`07_seq2seq_lstm.py`](07_seq2seq_lstm.py) | **Exp 7:** Encoder-Decoder architecture with LSTMs for sequence-to-sequence learning (sequence reversal). |
| [`08_rbm_binary.py`](08_rbm_binary.py) | **Exp 8:** Restricted Boltzmann Machine (RBM) for learning binary data representations on MNIST with Contrastive Divergence. |
| [`09_denoising_autoencoder.py`](09_denoising_autoencoder.py) | **Exp 9:** Denoising Convolutional Autoencoder to reconstruct clean images from noisy input data. |
| [`10_gan_mnist.py`](10_gan_mnist.py) | **Exp 10:** Deep Convolutional Generative Adversarial Network (DCGAN) on MNIST to generate handwritten digits. |

---

## 🚀 Requirements & Dependencies

Install the required Python packages before running the scripts:

```bash
pip install tensorflow numpy matplotlib
```

---

## 💻 How to Run

### Option 1: Direct Python Execution
Run any file directly from your terminal/command prompt:
```bash
python 01_xor_ffnn.py
python 04_cnn_mnist.py
```

### Option 2: Running in Jupyter Notebook
Open Jupyter Notebook in this directory:
```bash
jupyter notebook
```
Create a new Python 3 notebook and copy the code from any `.py` file directly into code cells and run.

---

## 📤 Push to GitHub

To push these programs to your GitHub repository, run the following git commands:

```bash
git init
git add .
git commit -m "Add CS722L5C Deep Learning TensorFlow lab programs"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```
