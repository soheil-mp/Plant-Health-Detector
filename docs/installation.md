# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 2GB+ free disk space for models and datasets

## Installation Steps

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

## Verifying Installation

To verify the installation:

1. Run the tests:
```bash
python -m unittest discover tests
```

2. Try running the prediction script:
```bash
predict-plant-disease --help
``` 