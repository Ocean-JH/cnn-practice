# CNN-Practice
A basic CNN framework for predicting dog emotions.



A Convolutional Neural Network (CNN) is a class of deep learning models specifically designed for processing structured grid data, such as images. CNNs are particularly effective for tasks involving image classification, object detection, and pattern recognition. The architecture of a CNN consists of several key layers, each serving a specific role in feature extraction and classification.

### 1. Convolutional Layer

The convolutional layer is the core building block of a CNN. It consists of a set of learnable filters (or kernels) that slide over the input image and perform convolution operations. The purpose of this layer is to automatically extract spatial features such as edges, textures, and patterns. Mathematically, the convolution operation is defined as:

$$(I∗K)(x,y)=∑i∑jI(x−i,y−j)K(i,j)(I * K)(x, y) = \sum_{i} \sum_{j} I(x - i, y - j) K(i, j)$$

where $I$ is the input image and $K$ is the filter kernel.

Each filter activates specific features in the input, and multiple filters allow the network to learn various representations of the data.

### 2. Activation Function (ReLU)

After each convolution operation, an activation function such as the Rectified Linear Unit (ReLU) is applied. ReLU introduces non-linearity into the network, allowing it to learn complex patterns. The function is defined as:

$$f(x)=max⁡(0,x)f(x) = \max(0, x)$$

This function replaces all negative pixel values with zero, enhancing the network's ability to model complex relationships in the data.

### 3. Pooling Layer

Pooling layers are used to reduce the spatial dimensions of feature maps while retaining the most important information. The most common type is max pooling, which selects the maximum value within a defined window (e.g., 2x2) and discards the rest. This helps in reducing computational complexity, controlling overfitting, and making the model more translation-invariant.

### 4. Batch Normalization

Batch normalization is applied to stabilize and accelerate training. It normalizes the activations of each layer, ensuring that inputs to subsequent layers maintain a stable distribution. This helps in reducing internal covariate shifts and improves convergence speed.

### 5. Dropout

Dropout is a regularization technique used to prevent overfitting. It randomly deactivates a fraction of neurons during training, forcing the network to learn more robust features. This improves generalization and ensures that the model does not rely on specific neurons too heavily.

### 6. Fully Connected Layers (Dense Layers)

After feature extraction by convolutional and pooling layers, the output is flattened into a one-dimensional vector and passed through one or more fully connected (dense) layers. These layers learn high-level representations and perform the final classification. The last dense layer contains a softmax activation function to output class probabilities:

$$P(yi)=ezi∑jezjP(y_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

where $ziz_i$ represents the logits for class $i$.

### 7. Model Compilation and Optimization

The model is compiled using the Adam optimizer, which is an adaptive gradient-based optimization algorithm that adjusts learning rates based on first and second-order moment estimates. The loss function used is categorical cross-entropy, given by:

$$L=−∑iyilog⁡(yi^)L = -\sum_{i} y_i \log(\hat{y_i})$$

where $yiy_i$ is the true label and $yi^\hat{y_i}$ is the predicted probability for class ii. This function measures the divergence between the true and predicted distributions.
