import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

class PlantDiseaseDetector:
    def __init__(self, model_path=None):
        """Initialize the Plant Disease Detector.
        
        Args:
            model_path (str, optional): Path to a pre-trained model. If None, uses default model.
        """
        self.model = None
        self.class_names = []
        self.image_size = (224, 224)  # Standard input size for most models
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load a pre-trained model.
        
        Args:
            model_path (str): Path to the model file
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_path):
        """Preprocess the input image for model inference.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        try:
            # Read and resize image
            image = Image.open(image_path)
            image = image.resize(self.image_size)
            
            # Convert to array and normalize
            image_array = np.array(image) / 255.0
            
            # Add batch dimension
            return np.expand_dims(image_array, axis=0)
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            raise
    
    def analyze_image(self, image_path):
        """Analyze a plant leaf image for diseases.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Analysis results containing disease, confidence, and treatment
        """
        if not self.model:
            raise ValueError("No model loaded. Please load a model first.")
            
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            
            # Make prediction
            predictions = self.model.predict(processed_image)
            
            # Get the predicted class and confidence
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # TODO: Replace with actual class names and treatments
            result = {
                "disease": f"Disease_{predicted_class}",
                "confidence": confidence,
                "treatment": "Please consult a plant expert for treatment options."
            }
            
            return result
        except Exception as e:
            print(f"Error analyzing image: {e}")
            raise 