"""
Task 2: Edge Detection - Classical vs Deep Learning
Implements: Sobel (manual), Canny, CNN
Dataset: BSD500 subset / Synthetic geometric shapes
"""

import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List

import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import utils
from utils import set_seed, setup_logging, save_metrics, save_plot, set_style


# ============================================================================
# Classical Edge Detection
# ============================================================================

class ClassicalEdgeDetection:
    """Manual implementation of Sobel edge detection."""

    @staticmethod
    def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Manual 2D convolution using NumPy.

        Steps:
          1. Zero-pad the image so output is the same spatial size as input.
          2. Flip the kernel (true convolution, not correlation).
          3. Slide the kernel over every (row, col) position.
          4. At each position: element-wise multiply patch × kernel, then sum.

        Args:
            image : 2-D float32 array  (H, W)
            kernel: 2-D float32 array  (kH, kW)

        Returns:
            output: 2-D float32 array  (H, W)
        """
        img_h, img_w = image.shape
        k_h, k_w = kernel.shape

        # Padding so output size matches input size  (same-padding)
        pad_h = k_h // 2  # e.g. 3//2 = 1  →  1 pixel border top & bottom
        pad_w = k_w // 2

        # Add zero-border around the image
        padded = np.pad(
            image,
            ((pad_h, pad_h), (pad_w, pad_w)),
            mode='constant',
            constant_values=0
        )

        # Flip kernel for mathematically correct convolution
        # (Sobel kernels are anti-symmetric so the result is identical,
        #  but flipping is the right thing to do in the general case)
        kernel_flipped = np.flipud(np.fliplr(kernel))

        # Output buffer
        output = np.zeros((img_h, img_w), dtype=np.float32)

        # Nested loop — slide the kernel over every pixel
        for row in range(img_h):
            for col in range(img_w):
                # Crop the patch under the kernel window
                patch = padded[row: row + k_h,
                        col: col + k_w]
                # Dot product: sum of element-wise products
                output[row, col] = np.sum(patch * kernel_flipped)

        return output
    @staticmethod
    def sobel_filter(image: np.ndarray) -> np.ndarray:
        """
        Manual Sobel edge detection.

        Pipeline:
          1. Define 3×3 Sobel kernels Kx (vertical edges) and Ky (horizontal).
          2. Convolve image with each kernel using the hand-written _convolve2d.
          3. Gradient magnitude  M = sqrt(Gx² + Gy²).
          4. Normalise to [0, 255] uint8 and return.

        Args:
            image: Grayscale image (H, W), uint8

        Returns:
            Edge magnitude map (H, W), uint8
        """
        img_f = image.astype(np.float32)

        # 3×3 Sobel kernels
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float32)  # detects vertical edges

        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=np.float32)  # detects horizontal edges

        # Fully manual convolution — no cv2.filter2D, no scipy
        Gx = ClassicalEdgeDetection._convolve2d(img_f, sobel_x)
        Gy = ClassicalEdgeDetection._convolve2d(img_f, sobel_y)

        # Combine gradients into a single magnitude map
        magnitude = np.sqrt(Gx ** 2 + Gy ** 2)

        # Normalise to 0–255
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max() * 255
        return magnitude.astype(np.uint8)

    
    @staticmethod
    def canny_edge(image: np.ndarray, low_thresh: int = 50, 
                   high_thresh: int = 150) -> np.ndarray:
        """
        Canny edge detection using OpenCV.
        
        Args:
            image: Grayscale image
            low_thresh: Lower threshold for hysteresis
            high_thresh: Upper threshold for hysteresis
            
        Returns:
            Binary edge map
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 1.4)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, low_thresh, high_thresh)
        
        return edges


# ============================================================================
# CNN for Edge Detection (TensorFlow/Keras)
# ============================================================================

def build_edge_cnn(input_shape=(256, 256, 1)):
    """
    Build CNN architecture for edge detection.
    
    Architecture: Encoder-Decoder style CNN
    - Encoder: Extract features with convolutions and pooling
    - Decoder: Upsample and reconstruct edge map
    
    Args:
        input_shape: Input image shape (H, W, C)
        
    Returns:
        Keras model
    """
    inputs = keras.Input(shape=input_shape)
    
    # Encoder
    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', 
                      name='conv1_1')(inputs)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', 
                      name='conv1_2')(x)
    x = layers.BatchNormalization()(x)
    pool1 = layers.MaxPooling2D((2, 2), name='pool1')(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', 
                      name='conv2_1')(pool1)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', 
                      name='conv2_2')(x)
    x = layers.BatchNormalization()(x)
    pool2 = layers.MaxPooling2D((2, 2), name='pool2')(x)
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', 
                      name='conv3_1')(pool2)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', 
                      name='conv3_2')(x)
    x = layers.BatchNormalization()(x)
    pool3 = layers.MaxPooling2D((2, 2), name='pool3')(x)
    
    # Bottleneck
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', 
                      name='bottleneck_1')(pool3)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', 
                      name='bottleneck_2')(x)
    x = layers.BatchNormalization()(x)
    
    # Decoder
    # Upsample 1
    x = layers.UpSampling2D((2, 2), name='upsample1')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', 
                      name='upconv1')(x)
    x = layers.BatchNormalization()(x)
    
    # Upsample 2
    x = layers.UpSampling2D((2, 2), name='upsample2')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', 
                      name='upconv2')(x)
    x = layers.BatchNormalization()(x)
    
    # Upsample 3
    x = layers.UpSampling2D((2, 2), name='upsample3')(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', 
                      name='upconv3')(x)
    x = layers.BatchNormalization()(x)
    
    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', 
                           name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='edge_cnn')
    
    return model


# ============================================================================
# Dataset Utilities
# ============================================================================

def create_synthetic_dataset(output_dir: Path, num_samples: int = 100):
    """
    Create synthetic edge detection dataset.
    Uses geometric shapes with known edges.
    
    Args:
        output_dir: Directory to save images
        num_samples: Number of samples to generate
    """
    logging.info(f"Creating synthetic edge dataset: {num_samples} samples")
    
    image_dir = output_dir / "images"
    edge_dir = output_dir / "edges"
    image_dir.mkdir(parents=True, exist_ok=True)
    edge_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_samples):
        # Create blank image
        img = np.ones((256, 256), dtype=np.uint8) * 200
        edge_gt = np.zeros((256, 256), dtype=np.uint8)
        
        # Random shapes
        num_shapes = np.random.randint(2, 5)
        
        for _ in range(num_shapes):
            shape_type = np.random.choice(['circle', 'rectangle', 'triangle'])
            
            if shape_type == 'circle':
                center = (np.random.randint(50, 206), np.random.randint(50, 206))
                radius = np.random.randint(20, 60)
                cv2.circle(img, center, radius, 100, -1)
                cv2.circle(edge_gt, center, radius, 255, 2)
            
            elif shape_type == 'rectangle':
                pt1 = (np.random.randint(20, 180), np.random.randint(20, 180))
                pt2 = (pt1[0] + np.random.randint(30, 80), 
                       pt1[1] + np.random.randint(30, 80))
                cv2.rectangle(img, pt1, pt2, 100, -1)
                cv2.rectangle(edge_gt, pt1, pt2, 255, 2)
            
            else:  # triangle
                pts = np.array([
                    [np.random.randint(50, 200), np.random.randint(50, 200)],
                    [np.random.randint(50, 200), np.random.randint(50, 200)],
                    [np.random.randint(50, 200), np.random.randint(50, 200)]
                ])
                cv2.fillPoly(img, [pts], 100)
                cv2.polylines(edge_gt, [pts], True, 255, 2)
        
        # Add noise
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Save
        cv2.imwrite(str(image_dir / f"{i:04d}.jpg"), img)
        cv2.imwrite(str(edge_dir / f"{i:04d}.png"), edge_gt)
    
    logging.info(f"Dataset created at {output_dir}")


def load_edge_dataset(data_dir: Path, split='train'):
    """
    Load edge detection dataset.
    
    Args:
        data_dir: Root directory of dataset
        split: 'train' or 'test'
        
    Returns:
        Tuple of (images, edges)
    """
    image_dir = data_dir / "images"
    edge_dir = data_dir / "edges"
    
    image_files = sorted(list(image_dir.glob("*.jpg")))
    edge_files = sorted(list(edge_dir.glob("*.png")))
    
    # Split data
    split_idx = int(0.8 * len(image_files))
    
    if split == 'train':
        image_files = image_files[:split_idx]
        edge_files = edge_files[:split_idx]
    else:
        image_files = image_files[split_idx:]
        edge_files = edge_files[split_idx:]
    
    images = []
    edges = []
    
    for img_path, edge_path in zip(image_files, edge_files):
        # Load and normalize
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        edge = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)
        
        img = img.astype(np.float32) / 255.0
        edge = (edge > 127).astype(np.float32)
        
        # Add channel dimension
        img = np.expand_dims(img, axis=-1)
        edge = np.expand_dims(edge, axis=-1)
        
        images.append(img)
        edges.append(edge)
    
    return np.array(images), np.array(edges)


# ============================================================================
# Edge Detection Pipeline
# ============================================================================

class EdgeDetectionPipeline:
    """Complete edge detection pipeline with classical and DL methods."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        set_seed(seed)
        tf.random.set_seed(seed)
        setup_logging("edge_detection.log")
        set_style()
        
        self.model = None
        self.classical_detector = ClassicalEdgeDetection()
        
        logging.info("EdgeDetectionPipeline initialized with TensorFlow")
    
    def prepare_data(self):
        """Prepare or download edge detection dataset."""
        data_dir = utils.DATA_DIR / "edge_data"
        
        if not data_dir.exists():
            create_synthetic_dataset(data_dir, num_samples=100)
        
        return data_dir
    
    def train_cnn(self, data_dir: Path, epochs: int = 30):
        """
        Train CNN for edge detection using TensorFlow.
        
        Args:
            data_dir: Root directory of dataset
            epochs: Number of training epochs
        """
        logging.info("Training CNN with TensorFlow/Keras...")
        
        # Load datasets
        X_train, y_train = load_edge_dataset(data_dir, 'train')
        X_val, y_val = load_edge_dataset(data_dir, 'test')
        
        logging.info(f"Train set: {X_train.shape}, Val set: {X_val.shape}")
        
        # Build model
        self.model = build_edge_cnn(input_shape=(256, 256, 1))
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), 
                    tf.keras.metrics.Recall()]
        )
        
        # Print model summary
        logging.info("\nModel Architecture:")
        self.model.summary(print_fn=logging.info)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=4,
            callbacks=callbacks,
            verbose=1
        )
        
        # Plot training curves
        self.plot_training_curves(history)
        
        logging.info("CNN training complete")
        
        return history
    
    def plot_training_curves(self, history):
        """Plot training and validation metrics."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(history.history['loss'], label='Train Loss')
        axes[0].plot(history.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Accuracy
        axes[1].plot(history.history['accuracy'], label='Train Accuracy')
        axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        save_plot(fig, "cnn_training_curve.png")
    
    def compare_methods(self, data_dir: Path):
        """
        Compare classical vs deep learning edge detection.
        
        Args:
            data_dir: Root directory of dataset
        """
        logging.info("Comparing edge detection methods...")
        
        # Load test images
        X_test, y_test = load_edge_dataset(data_dir, 'test')
        
        metrics_list = {'sobel': [], 'canny': [], 'cnn': []}
        
        # Process each test image
        for idx in range(len(X_test)):
            # Get image and ground truth
            img = (X_test[idx, :, :, 0] * 255).astype(np.uint8)
            edge_gt = y_test[idx, :, :, 0]
            
            # Sobel
            sobel_edges = self.classical_detector.sobel_filter(img)
            sobel_edges = (sobel_edges > 100).astype(np.float32)
            
            # Canny
            canny_edges = self.classical_detector.canny_edge(img)
            canny_edges = (canny_edges > 0).astype(np.float32)
            
            # CNN
            cnn_input = np.expand_dims(X_test[idx], axis=0)
            cnn_pred = self.model.predict(cnn_input, verbose=0)
            cnn_edges = (cnn_pred[0, :, :, 0] > 0.5).astype(np.float32)
            
            # Compute metrics
            for name, pred in [('sobel', sobel_edges), 
                              ('canny', canny_edges), 
                              ('cnn', cnn_edges)]:
                metrics = self.compute_metrics(edge_gt, pred)
                metrics_list[name].append(metrics)
        
        # Average metrics
        avg_metrics = {}
        for method in ['sobel', 'canny', 'cnn']:
            avg_metrics[method] = {
                'precision': np.mean([m['precision'] for m in metrics_list[method]]),
                'recall': np.mean([m['recall'] for m in metrics_list[method]]),
                'f1': np.mean([m['f1'] for m in metrics_list[method]]),
                'iou': np.mean([m['iou'] for m in metrics_list[method]])
            }
        
        # Log results
        logging.info("\nEdge Detection Comparison:")
        for method, metrics in avg_metrics.items():
            logging.info(f"\n{method.upper()}:")
            logging.info(f"  Precision: {metrics['precision']:.4f}")
            logging.info(f"  Recall: {metrics['recall']:.4f}")
            logging.info(f"  F1: {metrics['f1']:.4f}")
            logging.info(f"  IoU: {metrics['iou']:.4f}")
        
        # Save metrics
        save_metrics(avg_metrics, "edge_detection_comparison.json")
        
        # Visualize comparison (first test image)
        self.visualize_comparison(X_test[0], y_test[0])
        
        return avg_metrics
    
    @staticmethod
    def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
        """
        Compute precision, recall, F1, IoU for edge detection.
        
        Args:
            gt: Ground truth binary edge map
            pred: Predicted binary edge map
            
        Returns:
            Dictionary of metrics
        """
        gt = gt.flatten()
        pred = pred.flatten()
        
        tp = np.sum((gt == 1) & (pred == 1))
        fp = np.sum((gt == 0) & (pred == 1))
        fn = np.sum((gt == 1) & (pred == 0))
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        iou = tp / (tp + fp + fn + 1e-10)
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'iou': float(iou)
        }
    
    def visualize_comparison(self, img_array: np.ndarray, edge_gt: np.ndarray):
        """Create side-by-side comparison visualization."""
        # Prepare image
        img = (img_array[:, :, 0] * 255).astype(np.uint8)
        edge_gt_vis = (edge_gt[:, :, 0] * 255).astype(np.uint8)
        
        # Detect edges with all methods
        sobel = self.classical_detector.sobel_filter(img)
        canny = self.classical_detector.canny_edge(img)
        
        cnn_input = np.expand_dims(img_array, axis=0)
        cnn_pred = self.model.predict(cnn_input, verbose=0)
        cnn = (cnn_pred[0, :, :, 0] * 255).astype(np.uint8)
        
        # Plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(img, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(edge_gt_vis, cmap='gray')
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(sobel, cmap='gray')
        axes[0, 2].set_title('Sobel (Manual)')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(canny, cmap='gray')
        axes[1, 0].set_title('Canny')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(cnn, cmap='gray')
        axes[1, 1].set_title('CNN (Deep Learning)')
        axes[1, 1].axis('off')
        
        # Difference map (CNN vs Ground Truth)
        diff = np.abs(cnn.astype(float) - edge_gt_vis.astype(float))
        axes[1, 2].imshow(diff, cmap='hot')
        axes[1, 2].set_title('CNN Error Map')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        save_plot(fig, "edge_detection_comparison.png")
    
    def save_model(self):
        """Save trained CNN model."""
        model_path = utils.MODELS_DIR / "edge_cnn.h5"
        self.model.save(str(model_path))
        logging.info(f"Model saved to {model_path}")
    
    def export_to_onnx(self):
        """Export CNN to ONNX format."""
        try:
            import tf2onnx
            
            # Convert to ONNX
            spec = (tf.TensorSpec((None, 256, 256, 1), tf.float32, name="input"),)
            
            onnx_path = utils.MODELS_DIR / "edge_cnn.onnx"
            
            model_proto, _ = tf2onnx.convert.from_keras(
                self.model,
                input_signature=spec,
                opset=13,
                output_path=str(onnx_path)
            )
            
            logging.info(f"ONNX model saved to {onnx_path}")
            
            # Benchmark
            self.benchmark_onnx(onnx_path)
            
        except ImportError:
            logging.warning("tf2onnx not installed. Skipping ONNX export.")
        except Exception as e:
            logging.warning(f"ONNX export failed: {str(e)}")
    
    def benchmark_onnx(self, onnx_path: Path):
        """Benchmark TensorFlow vs ONNX inference latency."""
        try:
            import onnxruntime as ort
            
            # Load ONNX
            sess = ort.InferenceSession(str(onnx_path))
            
            # Prepare input
            dummy_input = np.random.randn(1, 256, 256, 1).astype(np.float32)
            
            # Benchmark TensorFlow
            tf_times = []
            for _ in range(100):
                start = time.time()
                _ = self.model.predict(dummy_input, verbose=0)
                tf_times.append(time.time() - start)
            
            # Benchmark ONNX
            onnx_times = []
            input_name = sess.get_inputs()[0].name
            for _ in range(100):
                start = time.time()
                _ = sess.run(None, {input_name: dummy_input})
                onnx_times.append(time.time() - start)
            
            tf_avg = np.mean(tf_times) * 1000
            onnx_avg = np.mean(onnx_times) * 1000
            speedup = tf_avg / onnx_avg
            
            logging.info(f"\nCNN Latency Benchmark:")
            logging.info(f"TensorFlow: {tf_avg:.2f} ms")
            logging.info(f"ONNX: {onnx_avg:.2f} ms")
            logging.info(f"Speedup: {speedup:.2f}x")
            
            benchmark_metrics = {
                'tensorflow_latency_ms': float(tf_avg),
                'onnx_latency_ms': float(onnx_avg),
                'speedup': float(speedup)
            }
            save_metrics(benchmark_metrics, "cnn_onnx_benchmark.json")
            
        except Exception as e:
            logging.warning(f"ONNX benchmarking failed: {str(e)}")
    
    def run_pipeline(self):
        """Execute complete edge detection pipeline."""
        logging.info("="*60)
        logging.info("TASK 2: EDGE DETECTION PIPELINE (CNN + TensorFlow)")
        logging.info("="*60)
        
        # 1. Prepare data
        data_dir = self.prepare_data()
        
        # 2. Train CNN
        self.train_cnn(data_dir, epochs=30)
        
        # 3. Compare methods
        metrics = self.compare_methods(data_dir)
        
        # 4. Save model
        self.save_model()
        
        # 5. Export to ONNX
        self.export_to_onnx()
        
        logging.info("="*60)
        logging.info("TASK 2 COMPLETE")
        logging.info("="*60)
        
        return metrics


if __name__ == "__main__":
    pipeline = EdgeDetectionPipeline()
    metrics = pipeline.run_pipeline()
    
    print("\n" + "="*60)
    print("EDGE DETECTION RESULTS (CNN with TensorFlow)")
    print("="*60)
    for method, m in metrics.items():
        print(f"\n{method.upper()}:")
        print(f"  F1: {m['f1']:.4f}, IoU: {m['iou']:.4f}")
    print("="*60)
