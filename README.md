# Plant Health Detector 🌿

A deep learning-based plant disease detection system that helps identify plant diseases from leaf images.

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

## Features

- 🔍 Real-time plant disease detection
- 🌿 Support for multiple plant species
- 📊 Confidence scores for predictions
- 💡 Treatment recommendations
- 🚀 Easy-to-use command-line interface

## Project Structure

```
Plant-Health-Detector/
├── data/               # Data directory
│   ├── raw/           # Raw training data
│   └── processed/     # Processed data
├── docs/              # Documentation
├── models/            # Trained models
├── src/               # Source code
│   ├── plant_detector/
│   ├── train.py
│   └── predict.py
└── tests/             # Test files
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 2GB+ free disk space

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Plant-Health-Detector.git
cd Plant-Health-Detector
```

2. Create and activate a virtual environment:
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Unix/MacOS
python -m venv venv
source venv/bin/activate
```

3. Install the package:
```bash
pip install -e .
```

## Usage

### Training a Model

To train a new model:

```bash
train-plant-detector --data_dir data/raw --epochs 50
```

Options:
- `--data_dir`: Directory containing training data
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 32)
- `--output_dir`: Directory to save models (default: models/)

### Making Predictions

For a single image:
```bash
predict-plant-disease --model_path models/best_model.h5 --image path/to/image.jpg
```

For batch processing:
```bash
predict-plant-disease --model_path models/best_model.h5 --input_dir path/to/images --output_dir results
```

## Model Architecture

The system uses a ResNet50-based architecture with the following modifications:
- Pre-trained weights from ImageNet
- Additional dense layers for classification
- Dropout for regularization
- Data augmentation during training

## Development

### Running Tests

```bash
python -m unittest discover tests
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ResNet architecture by Microsoft Research
- Plant Village Dataset
- TensorFlow and OpenCV communities
