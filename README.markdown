# Autoencoder for MNIST Image Reconstruction

## Overview
This project implements autoencoders using PyTorch to reconstruct images from the MNIST dataset. Two autoencoder models are included: a linear autoencoder (`Autoencoder_Linear`) and a convolutional autoencoder (`Autoencoder`). The goal is to compress MNIST digit images into a lower-dimensional latent space and reconstruct them, demonstrating unsupervised learning for image processing. The project includes data preprocessing, model training, and visualization of original and reconstructed images.

## Features
- **Dataset**: MNIST dataset (60,000 training images of handwritten digits, 28x28 pixels).
- **Models**:
  - **Linear Autoencoder**: Compresses images to a 3D latent space using fully connected layers.
  - **Convolutional Autoencoder**: Uses convolutional layers to encode images into a 64-channel latent representation.
- **Preprocessing**: Images are converted to tensors with pixel values in [0, 1].
- **Training**: Uses Mean Squared Error (MSE) loss and Adam optimizer.
- **Visualization**: Displays original and reconstructed images for qualitative evaluation.
- **Performance**: Reconstructs MNIST digits with high fidelity (qualitative results visualized).

## Prerequisites
- Python 3.7+
- PyTorch
- torchvision
- Matplotlib
- Conda or virtualenv
- CUDA-enabled GPU (optional, for faster training)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Sriram1918/Sriram1918-sentiment-analysis-with-deep-neural-networks.git
   cd Sriram1918-sentiment-analysis-with-deep-neural-networks
   ```

2. **Set Up Environment**:
   Using Conda:
   ```bash
   conda env create -f conda_env.yml
   conda activate autoencoder_env
   ```
   Or using virtualenv:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Install Dependencies**:
   Ensure the following are installed (included in `requirements.txt`):
   ```bash
   pip install torch torchvision matplotlib
   ```

4. **Download Dataset**:
   The MNIST dataset is automatically downloaded to the `./data` directory when running the notebook.

## Dataset
- **Source**: MNIST dataset, containing 60,000 training images (28x28 grayscale handwritten digits).
- **Preprocessing**:
  - Images are loaded using `torchvision.datasets.MNIST`.
  - Transformed to tensors with `transforms.ToTensor()` (pixel values in [0, 1]).
  - Batched with a size of 64 for training.

## Model Architecture
1. **Linear Autoencoder (`Autoencoder_Linear`)**:
   - **Encoder**: 
     - Input: Flattened 28x28 images (784 dimensions).
     - Layers: Linear(784→128) → ReLU → Linear(128→64) → ReLU → Linear(64→12) → ReLU → Linear(12→3).
     - Output: 3D latent space.
   - **Decoder**:
     - Input: 3D latent vector.
     - Layers: Linear(3→12) → ReLU → Linear(12→64) → ReLU → Linear(64→128) → ReLU → Linear(128→784) → Sigmoid.
     - Output: Reconstructed 28x28 image.

2. **Convolutional Autoencoder (`Autoencoder`)**:
   - **Encoder**:
     - Input: 1x28x28 images (1 channel).
     - Layers: Conv2d(1→16, 3x3, stride=2) → ReLU → Conv2d(16→32, 3x3, stride=2) → ReLU → Conv2d(32→64, 7x7).
     - Output: 64x1x1 latent representation.
   - **Decoder**:
     - Input: 64x1x1 latent representation.
     - Layers: ConvTranspose2d(64→32, 7x7) → ReLU → ConvTranspose2d(32→16, 3x3, stride=2) → ReLU → ConvTranspose2d(16→1, 3x3, stride=2) → Sigmoid.
     - Output: Reconstructed 1x28x28 image.

## Usage
1. **Run the Notebook**:
   - Open `Autoencoder.ipynb` in Jupyter Notebook or JupyterLab.
   - Execute cells sequentially to:
     - Load and preprocess the MNIST dataset.
     - Define and initialize the autoencoder models.
     - Train the convolutional autoencoder (or modify to train the linear one).
     - Visualize reconstructed images.

2. **Training**:
   - The notebook trains the convolutional autoencoder using:
     - Loss: Mean Squared Error (MSE).
     - Optimizer: Adam (learning rate = 1e-3, weight decay = 1e-5).
     - Batch size: 64.
   - To train the linear autoencoder, modify the model instantiation to `model = Autoencoder_Linear()` and flatten input images.

3. **Visualization**:
   - The notebook includes code to plot original and reconstructed images every 4 epochs.
   - Outputs are displayed using Matplotlib, showing 9 images per epoch.

   Example visualization code:
   ```python
   plt.figure(figsize=(9, 2))
   plt.gray()
   for i, item in enumerate(imgs):
       if i >= 9: break
       plt.subplot(2, 9, i+1)
       plt.imshow(item[0])
   for i, item in enumerate(recon):
       if i >= 9: break
       plt.subplot(2, 9, 9+i+1)
       plt.imshow(item[0])
   ```

## Project Structure
```
├── data/
│   ├── MNIST/              # Automatically downloaded MNIST dataset
├── notebooks/
│   ├── Autoencoder.ipynb   # Main notebook with code and visualizations
├── requirements.txt        # Python dependencies
├── conda_env.yml          # Conda environment configuration
└── README.md              # This file
```

## Results
- **Qualitative**: Visualizations show that both autoencoders reconstruct MNIST digits with reasonable fidelity, with the convolutional autoencoder typically producing sharper images.
- **Quantitative**: The notebook uses MSE loss to measure reconstruction error (specific values depend on training duration and model).
- **Training Time**: Approximately 5-10 minutes per epoch on a CPU, faster with a GPU.

## Notes
- The notebook includes a comment about using `nn.MaxPool2d` and inspecting encoded data as homework, suggesting potential extensions.
- The linear autoencoder requires flattened inputs (`item.reshape(-1, 28, 28)`), while the convolutional autoencoder processes 2D images directly.
- The dataset is not normalized to [-1, 1] as suggested by the `nn.Tanh` comment; instead, `nn.Sigmoid` is used with [0, 1] inputs.

## Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## References
- LeCun, Y., et al. "The MNIST database of handwritten digits." 1998.
- Goodfellow, I., et al. "Deep Learning." MIT Press, 2016.
- PyTorch Documentation: [pytorch.org](https://pytorch.org).

## License
MIT License. See `LICENSE` for details.