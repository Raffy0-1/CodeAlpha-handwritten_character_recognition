# Handwritten Character Recognition

A deep learning project that builds a complete handwritten character recognition pipeline from scratch.
The project covers digit recognition on MNIST, full character recognition on EMNIST, model interpretability
using Grad-CAM, and a conceptual walkthrough of sequence-based word recognition using CRNN with CTC loss.

Built and trained on Google Colab using a free T4 GPU.

---

## Project Overview

The goal of this project is not just to train a model and report accuracy. Every design decision
is explained — from why pixel values are normalized, to how Grad-CAM computes gradients, to why
CTC loss is necessary for word-level recognition without character segmentation.

The project is structured as a progressive learning path:

- Start with MNIST (10 classes, digits only) to understand the CNN pipeline end to end
- Scale to EMNIST (47 classes, digits and letters) using the exact same architecture
- Apply Grad-CAM to visualize what the trained model actually learned to look at
- Study the CRNN architecture and understand how it extends character recognition to full words

---

## Results

| Dataset | Classes | Test Accuracy | Test Loss |
|---------|---------|---------------|-----------|
| MNIST   | 10      | 99.43%        | 0.0187    |
| EMNIST  | 47      | 89.51%        | 0.2980    |

The EMNIST accuracy is lower than MNIST as expected. With 47 classes, many character pairs are
visually ambiguous in handwriting. The most frequently confused pairs observed during evaluation:

- F confused with f: 167 times
- L confused with 1: 161 times
- O confused with 0: 159 times
- I confused with 1: 124 times

These confusions are inherent to the dataset, not flaws in the model. A human reader would make
the same errors without surrounding word context.

---

## Model Architecture

Both the MNIST and EMNIST models use the same CNN architecture. Only the output layer changes.

```
Input (28, 28, 1)

Convolutional Block 1
  Conv2D(32, 3x3, padding=same, relu)
  BatchNormalization
  Conv2D(32, 3x3, padding=same, relu)
  BatchNormalization
  MaxPooling2D(2x2)          -> (14, 14, 32)
  Dropout(0.25)

Convolutional Block 2
  Conv2D(64, 3x3, padding=same, relu)
  BatchNormalization
  Conv2D(64, 3x3, padding=same, relu)
  BatchNormalization
  MaxPooling2D(2x2)          -> (7, 7, 64)
  Dropout(0.25)

Fully Connected Head
  Flatten                    -> (3136,)
  Dense(256, relu)
  BatchNormalization
  Dropout(0.50)
  Dense(num_classes, softmax)

Optimizer : Adam (lr=0.001)
Loss      : Categorical Crossentropy
```

---

## Grad-CAM

After training, Grad-CAM is applied to every test digit to visualize which regions of the image
the model focused on when making its prediction.

Grad-CAM works by computing the gradient of the predicted class score with respect to the feature
maps of the last convolutional layer. These gradients are averaged across spatial dimensions to
produce an importance weight per filter. A weighted sum of the feature maps is then upsampled and
overlaid on the original image as a heatmap.

This confirms that the model is genuinely reading digit strokes rather than memorizing background
patterns or image artifacts.

---

## CRNN Architecture (Conceptual)

The CRNN section demonstrates how to extend character-level recognition to full word recognition
without needing to segment individual characters first.

```
Input: Word image (32, 128, 1)

Stage 1 - Convolutional Feature Extraction
  Conv Block 1  : (32, 128, 1)   ->  (16, 64, 32)
  Conv Block 2  : (16, 64, 32)   ->  (8,  32, 64)
  Conv Block 3  : (8,  32, 64)   ->  (8,  32, 128)

Stage 2 - Sequence Conversion
  Reshape       : (8, 32, 128)   ->  (32, 1024)
  Dense bridge  : (32, 1024)     ->  (32, 64)

Stage 3 - Recurrent Modeling
  BiLSTM        : (32, 64)       ->  (32, 256)
  BiLSTM        : (32, 256)      ->  (32, 128)

Output: (32, 80) - probability over 80 tokens at each of 32 time steps
        (79 printable ASCII characters + 1 CTC blank token)

Decoding: CTC beam search collapses repeated characters and blank tokens into a word string
```

To train the CRNN you need a word-level dataset such as the IAM Handwriting Database.
The architecture code is fully implemented and ready for that dataset to be plugged in.

---

## Project Structure

```
handwriting_recognition/
    handwritten_character_recognition.ipynb   main notebook

On Google Drive (auto-created during training):
    handwriting_recognition/
        models/
            mnist_cnn.h5                      trained MNIST weights  (10.07 MB)
            emnist_cnn.h5                     trained EMNIST weights
        history/
            mnist_training_log.csv            epoch-by-epoch metrics
            emnist_training_log.csv
            mnist_history.pkl
            emnist_history.pkl
        datasets/
            emnist_balanced.npz               cached dataset to avoid re-downloading
```

---

## Setup and Usage

### Running on Google Colab (recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload the notebook: File -> Upload notebook
3. Set the runtime to GPU: Runtime -> Change runtime type -> T4 GPU
4. Run all cells from top to bottom

On the first run, training takes roughly 15-20 minutes for MNIST and 25-35 minutes for EMNIST
depending on GPU availability. All weights and logs are saved to your Google Drive automatically.

On any subsequent session reset, the notebook detects saved weights on Drive and skips retraining.
Only the Drive mount cell and the imports cell need to run before loading the model.

### Running locally

```bash
git clone https://github.com/your-username/handwriting-recognition.git
cd handwriting-recognition
pip install -r requirements.txt
jupyter notebook handwritten_character_recognition.ipynb
```

When running locally, comment out the Google Drive mount cell (Section 1) and update
the path constants to local directories of your choice.

---

## Dependencies

| Package                  | Purpose                                      |
|--------------------------|----------------------------------------------|
| tensorflow 2.20.0        | Model building, training, Keras API          |
| tensorflow-datasets      | EMNIST dataset download and loading          |
| numpy                    | Array operations                             |
| opencv-python-headless   | Image resizing and colormap operations       |
| matplotlib               | Visualization and plotting                   |
| seaborn                  | Confusion matrix heatmaps                    |
| scikit-learn             | Metrics: confusion matrix, classification report |
| pandas                   | Reading training log CSV files               |

Full pinned versions are in `requirements.txt`.

---

## Key Concepts Covered

**Why normalize pixel values**
Raw pixel values (0-255) cause large gradient updates during backpropagation. Scaling to [0, 1]
keeps weight updates stable and speeds up convergence.

**Why BatchNormalization**
Normalizes layer outputs to zero mean and unit variance. Prevents internal covariate shift,
which is what happens when the distribution of inputs to a layer shifts during training.

**Why Dropout**
Randomly disables neurons during training. Forces the network to learn distributed representations
rather than relying on specific neurons, which reduces overfitting.

**Why CTC loss for word recognition**
A CNN outputs one prediction per fixed-size image. A word image contains multiple characters with
no clear boundaries. CTC loss allows training with word-level labels only, without requiring
the model to know where each character starts or ends.

---

## Extending the Project

A few directions if you want to take this further:

- Train the CRNN on the IAM Handwriting Database for real word recognition
- Add data augmentation (rotation, elastic distortion) to make the CNN more robust
- Replace the CNN backbone with a lightweight pretrained model like MobileNetV2
- Build a Gradio or Streamlit interface so users can draw a character and get a prediction
- Export the trained model to TFLite for mobile deployment

---

## References

- LeCun et al. (1998). Gradient-based learning applied to document recognition.
- Cohen et al. (2017). EMNIST: Extending MNIST to handwritten letters.
- Selvaraju et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization.
- Shi et al. (2016). An end-to-end trainable neural network for image-based sequence recognition.
- IAM Handwriting Database: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
