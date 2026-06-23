# China City Classification and Scene Density Estimation with TensorFlow

## Overview

This project implements a **multi-task deep learning pipeline** using TensorFlow and Keras for aerial scene understanding on the **VisDrone dataset**.

The model simultaneously performs:

1. **City Classification** – predicts cluster-based city labels generated through unsupervised learning.
2. **Scene Density Estimation** – predicts human and vehicle density levels from aerial imagery.

Instead of relying on manually labeled city annotations, the project generates pseudo-labels using **MobileNetV2 feature extraction**, **PCA dimensionality reduction**, and **MiniBatch K-Means clustering**.

The architecture leverages scene density information as contextual cues for improving city classification performance.

---

## Features

- Multi-task deep learning architecture
- Transfer learning with ImageNet-pretrained MobileNetV2
- Human and vehicle scene density estimation
- Automatic city label generation using PCA + MiniBatch K-Means
- Feature fusion between density predictions and city classification
- Gradient isolation using `tf.stop_gradient()`
- Efficient TensorFlow data pipelines
- Cosine learning-rate scheduling
- CPU-friendly training
- Support for train, validation, and test datasets

---

## Project Structure

```text
china.ipynb
requirements.txt
│
├── Dataset Loading
├── Image Preprocessing
├── Density Preprocessing
├── City Label Generation (PCA + MiniBatch K-Means)
├── TensorFlow Pipelines
├── Multi-Head Model Definition
└── Training Configuration
```

---

## Model Architecture

### Shared Backbone

- MobileNetV2 (ImageNet pretrained)
- Frozen feature extractor
- Input resolution: **224 × 407**

### Density Head

Predicts:

- Human density
- Vehicle density

Human categories:

- Pedestrian
- People

Vehicle categories:

- Car
- Van
- Truck
- Bus
- Motorcycle

To stabilize training and reduce the influence of extreme crowd counts, density targets are transformed using:

```python
counts_vector = tf.math.log1p(counts_vector)
```

This allows the network to learn scene density rather than exact object counts.

---

### City Classification Head

Predicts one of **14 cluster-based city categories** generated through PCA and MiniBatch K-Means:

- Tianjin
- Hong Kong
- Daqing
- Ganzhou
- Guangzhou
- Jinchang
- Liuzhou
- Nanjing
- Shaoxing
- Shenyang
- Nanyang
- Zhangjiakou
- Suzhou
- Xuzhou

The classification branch incorporates density predictions as additional contextual information.

```python
protected_counts = tf.stop_gradient(count_output)
```

This prevents classification gradients from interfering with density learning.

---

## Feature Fusion

The model combines:

1. Global image features extracted from MobileNetV2.
2. Predicted human density.
3. Predicted vehicle density.

These features are concatenated before the final city classification layers, enabling the model to use scene activity information when distinguishing between city clusters.

---

## Dataset

The project uses the **VisDrone dataset** loaded through Deep Lake:

```python
train_ds = dl.load("hub://activeloop/visdrone-det-train")
val_ds = dl.load("hub://activeloop/visdrone-det-val")
test_ds = dl.load("hub://activeloop/visdrone-det-test-dev")
```

---

## Image Preprocessing

Images are:

1. Resized to **224 × 407**
2. Normalized using MobileNetV2 preprocessing

```python
image = tf.image.resize(image, (224, 407))
image = keras.applications.mobilenet_v2.preprocess_input(image)
```

---

## City Label Generation Pipeline

City labels are generated automatically by:

1. Extracting MobileNetV2 features
2. Applying Global Average Pooling
3. Reducing dimensions with PCA

```python
PCA(n_components=50)
```

4. Clustering features using MiniBatch K-Means

```python
MiniBatchKMeans(
    n_clusters=14,
    batch_size=1024
)
```

---

## Training Configuration

### Batch Size

```python
BATCH_SIZE = 16
```

### Optimizer

- Adam optimizer
- CosineDecay learning-rate scheduler

### Loss Functions

#### City Classification

```python
SparseCategoricalCrossentropy
```

#### Scene Density Estimation

```python
Huber Loss
```

---

## Results

Final evaluation metrics:

```text
city_output_accuracy: 0.8441
city_output_loss:     0.3893

count_output_loss:    0.4361
count_output_mae:     0.8057

total_loss:           0.6376
```

### Performance Summary

- **City Classification Accuracy:** **84.41%**
- **Human/Vehicle Density MAE:** **0.81**
- Stable multi-task training
- Successful integration of density cues into classification

---

## Technologies Used

- TensorFlow
- Keras
- MobileNetV2
- Deep Lake
- Scikit-learn
  - PCA
  - MiniBatch K-Means

---

## Future Improvements

- Fine-tune the MobileNetV2 backbone
- Add data augmentation
- Incorporate additional object categories
- Explore attention mechanisms
- Export the model for deployment
- Evaluate with additional metrics

---

## License

This project is provided for educational and research purposes under the **MIT License**.
