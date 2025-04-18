
# Offline Signature Forgery Detection (MSLG-Net)

This repository provides the complete implementation of **MSLG-Net**: a deep learning model for robust offline signature verification using Multi-Scale Feature Attention (MSFA) and Local-Global Feature Integration (LGFI).

## 📌 Features
- Multi-scale attention mechanism for fine-to-coarse feature extraction
- Local-global fusion block to integrate structural and contextual cues
- Hyperband optimization for efficient hyperparameter search
- Adversarial robustness testing (FGSM, PGD, JPEG, Gaussian noise)
- Comparison with pretrained models (VGG16, ResNet50, etc.)
- Full ablation study for component-level contribution analysis

## 🗂️ Directory Structure
```
├── dataset_preparation.py         # Load and preprocess datasets
├── model_mslgnet.py               # Define MSLG-Net architecture
├── train_mslgnet.py               # Train the MSLG-Net model
├── evaluate_mslgnet.py            # Evaluate model performance
├── robustness_evaluation.py       # Evaluate adversarial and noisy perturbations
├── ablation_study.py              # Run ablation variants (no MSFA/LGFI/single MSFA)
├── compare_pretrained_models.py   # Benchmark against VGG16, ResNet50, etc.
├── visualize_results.py           # Plot accuracy and confusion matrices
├── results_metrics.txt            # Stores accuracy, precision, recall, F1-score
├── ablation_results.txt           # Stores ablation study results
├── pretrained_comparison_results.txt  # Metrics from pretrained model benchmarks
```

## 🧪 Datasets
- CEDAR
- UTSig
- BHSig260-Bengali
- BHSig260-Hindi

Organize datasets into subfolders: `datasets/<dataset_name>/genuine` and `forged`

## 🚀 Quick Start

1. Install required libraries:
```bash
pip install tensorflow scikit-learn opencv-python matplotlib seaborn
```

2. Prepare the data:
```bash
python dataset_preparation.py
```

3. Train the model:
```bash
python train_mslgnet.py
```

4. Evaluate the model:
```bash
python evaluate_mslgnet.py
```

5. Run adversarial tests:
```bash
python robustness_evaluation.py
```

6. Run ablation studies:
```bash
python ablation_study.py
```

7. Compare with pretrained models:
```bash
python compare_pretrained_models.py
```

8. Plot the results:
```bash
python visualize_results.py
```

## 📄 License
This project is licensed under the MIT License.
