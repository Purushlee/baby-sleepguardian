# Baby Cry Detection ML

## Overview
This project detects baby cries and classifies cry types (hungry, sleepy, pain, discomfort).

## Folder Structure
- `dataset/` - Audio files for training/testing
- `models/` - Saved ML models
- `features/` - Feature extraction scripts
- `training/` - Training scripts
- `inference/` - Prediction scripts
- `logs/` - Training logs
- `results/` - Evaluation outputs

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Train models: `python training/train_binary.py` or `train_multi_class.py`
3. Run inference: `python inference/predict.py`

