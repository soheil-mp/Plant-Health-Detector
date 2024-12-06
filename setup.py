from setuptools import setup, find_packages

# Read README with proper encoding
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="plant-health-detector",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tensorflow>=2.0.0",
        "opencv-python>=4.0.0",
        "numpy>=1.19.0",
        "pandas>=1.0.0",
        "scikit-learn>=0.24.0",
        "Pillow>=8.0.0",
        "matplotlib>=3.3.0",
    ],
    entry_points={
        "console_scripts": [
            "train-plant-detector=train:main",
            "predict-plant-disease=predict:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A plant disease detection system using computer vision and deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Plant-Health-Detector",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.8",
) 