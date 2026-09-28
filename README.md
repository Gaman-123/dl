# CS722L5C Deep Learning Lab Programs

This repository contains TensorFlow & Keras implementations of all 10 programs specified in the **CS722L5C Deep Learning Lab Manual** (Department of Computer Science & Engineering, SCEM, Mangaluru).

Both `.py` scripts and Jupyter Notebook (`.ipynb`) files are provided for every program.

---

## 📁 Repository Structure

| Experiment | Python Script (`.py`) | Jupyter Notebook (`.ipynb`) | Description |
| :--- | :--- | :--- | :--- |
| **Exp 1** | [`01_xor_ffnn.py`](01_xor_ffnn.py) | [`01_xor_ffnn.ipynb`](01_xor_ffnn.ipynb) | Deep feedforward neural network to learn XOR function. |
| **Exp 2** | [`02_regularization_mnist.py`](02_regularization_mnist.py) | [`02_regularization_mnist.ipynb`](02_regularization_mnist.ipynb) | Regularization techniques (L1/L2 penalties, Data Augmentation, Gaussian Noise, Dropout, EarlyStopping). |
| **Exp 3** | [`03_optimizers_comparison.py`](03_optimizers_comparison.py) | [`03_optimizers_comparison.ipynb`](03_optimizers_comparison.ipynb) | Comparison of optimization algorithms (SGD, Momentum, Adam) on XOR. |
| **Exp 4** | [`04_cnn_mnist.py`](04_cnn_mnist.py) | [`04_cnn_mnist.ipynb`](04_cnn_mnist.ipynb) | Convolutional Neural Network (CNN) on MNIST digit classification. |
| **Exp 5** | [`05_lstm_text_classification.py`](05_lstm_text_classification.py) | [`05_lstm_text_classification.ipynb`](05_lstm_text_classification.ipynb) | LSTM-based RNN for text classification on IMDB dataset. |
| **Exp 6** | [`06_bidirectional_rnn.py`](06_bidirectional_rnn.py) | [`06_bidirectional_rnn.ipynb`](06_bidirectional_rnn.ipynb) | Bidirectional RNN vs Standard SimpleRNN comparison on sequence prediction. |
| **Exp 7** | [`07_seq2seq_lstm.py`](07_seq2seq_lstm.py) | [`07_seq2seq_lstm.ipynb`](07_seq2seq_lstm.ipynb) | Encoder-Decoder architecture with LSTMs for sequence-to-sequence learning. |
| **Exp 8** | [`08_rbm_binary.py`](08_rbm_binary.py) | [`08_rbm_binary.ipynb`](08_rbm_binary.ipynb) | Restricted Boltzmann Machine (RBM) for binary data representations on MNIST. |
| **Exp 9** | [`09_denoising_autoencoder.py`](09_denoising_autoencoder.py) | [`09_denoising_autoencoder.ipynb`](09_denoising_autoencoder.ipynb) | Denoising Convolutional Autoencoder to reconstruct clean images. |
| **Exp 10** | [`10_gan_mnist.py`](10_gan_mnist.py) | [`10_gan_mnist.ipynb`](10_gan_mnist.ipynb) | Deep Convolutional Generative Adversarial Network (DCGAN) on MNIST. |

---

## 🚀 Requirements & Dependencies

Install the required Python packages before running:

```bash
pip install tensorflow numpy matplotlib jupyter
```

---

## 💻 How to Run

### Option 1: Jupyter Notebook (`.ipynb`)
Launch Jupyter Notebook in this directory:
```bash
jupyter notebook
```
Open any `.ipynb` file and execute the cells.

### Option 2: Direct Python Execution (`.py`)
Run any script directly from terminal:
```bash
python 01_xor_ffnn.py
python 04_cnn_mnist.py
```
