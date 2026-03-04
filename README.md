# Pneumonia_AI_lung_auscultation_sounds

Pneumonia Detection using Lung Auscultation Sounds
Overview

This project develops a deep learning model to detect pneumonia from lung auscultation sounds using Mel-spectrogram based audio classification.
The pipeline converts respiratory sound recordings into Mel-spectrogram images and trains a convolutional neural network (CNN) to classify Healthy vs Pneumonia patients.

The system is built using the Respiratory Sound Database (ICBHI dataset) and applies audio preprocessing, augmentation, and CNN-based image classification.

Dataset

The dataset used in this project is the Respiratory Sound Database (ICBHI 2017 dataset).

This dataset contains respiratory recordings collected using digital stethoscopes and microphones from multiple patients.

Each audio file corresponds to a patient and is associated with a diagnosis.

For this project, patient-level labels were used to construct a binary classification dataset:

Healthy
Pneumonia

The dataset was organized into the following structure:

MelSpectrogram_Dataset_Split
│
├── train
│   ├── Healthy
│   └── Pneumonia
│
└── test
    ├── Healthy
    └── Pneumonia

To avoid data leakage, the dataset was split based on patient IDs, ensuring that recordings from the same patient do not appear in both the training and testing sets.

Data Preprocessing

The respiratory audio recordings were converted into Mel-spectrogram images through several preprocessing steps.

Audio Processing Pipeline
Respiratory sound (.wav)
        ↓
Resampling (16 kHz)
        ↓
Bandpass Filtering (50–2000 Hz)
        ↓
Length Normalization (5 seconds)
        ↓
Mel-Spectrogram Extraction
        ↓
Log Scaling
        ↓
Normalization
        ↓
Image Generation (224 × 224)
Key Parameters
Parameter	Value
Sampling Rate	16000 Hz
FFT Size	1024
Hop Length	256
Mel Bands	128
Frequency Range	50–2000 Hz

A bandpass filter (50–2000 Hz) was applied to focus on the frequency range where lung sounds such as crackles and wheezes occur.

Data Augmentation

To increase the training data diversity, time-shift augmentation was applied to the audio signals before generating Mel-spectrograms.

The augmentation process includes:

Audio time shifting (±1.5 seconds)

This simulates variations in respiratory events occurring at different temporal positions within the recording.

Augmentation was applied only to training data to prevent bias in model evaluation.

Mel-Spectrogram Generation

The Mel-spectrograms were generated using the librosa library and converted into image representations suitable for CNN training.

Each spectrogram image was normalized and saved as a 224 × 224 RGB image using a perceptually optimized color map.

Model Architecture

A ResNet18 convolutional neural network pretrained on ImageNet was used for classification.

The final classification layer was modified to support binary classification.

Mel Spectrogram Image (224×224)
        ↓
ResNet18 (ImageNet pretrained)
        ↓
Dropout
        ↓
Fully Connected Layer
        ↓
Softmax
        ↓
Healthy / Pneumonia
Training Strategy

The model was trained using the following settings:

Parameter	Value
Batch Size	64
Optimizer	AdamW
Learning Rate	2.5e-4
Weight Decay	1e-4
Epochs	10
Loss Function	CrossEntropyLoss
Label Smoothing	0.05

To ensure reproducibility and robust evaluation, the model was trained using multiple random seeds.

Evaluation Metrics

The model performance was evaluated using multiple metrics:

Accuracy

Precision

Recall

F1-score

ROC AUC

Confusion Matrix

Confusion matrices and classification reports were generated to analyze class-wise performance.

Evaluation Pipeline
Test Mel-spectrogram images
        ↓
ResNet18 inference
        ↓
Prediction probabilities
        ↓
Metrics computation
        ↓
Confusion matrix visualization
Results

Example evaluation output includes:

Confusion matrix

Precision / Recall / F1-score per class

ROC AUC score

These metrics help analyze how well the model detects pneumonia cases from respiratory sounds.

Project Structure
Pneumonia_AI_lung_auscultation_sounds
│
├── data_preprocessing
│   ├── audio_augmentation.py
│   ├── mel_spectrogram_generation.py
│
├── training
│   ├── lung_sound_train.py
│
├── evaluation
│   ├── model_test.py
│
├── results
│   ├── checkpoints
│   ├── confusion_matrix
│
└── README.md
Future Work

Future improvements may include:

Incorporating breathing cycle segmentation

Using transformer-based audio models

Applying advanced audio augmentation methods

Expanding the dataset for improved generalization

Conclusion

This project demonstrates an end-to-end pipeline for pneumonia detection using lung auscultation sounds and Mel-spectrogram based deep learning.

The approach leverages audio preprocessing, data augmentation, and CNN-based classification to detect respiratory abnormalities from stethoscope recordings.