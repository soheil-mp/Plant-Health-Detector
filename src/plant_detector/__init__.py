import os
import cv2
import numpy as np
import tensorflow as tf
import json
from PIL import Image

class PlantDiseaseDetector:
    def __init__(self, model_path=None):
        """Initialize the Plant Disease Detector.
        
        Args:
            model_path (str, optional): Path to a pre-trained model. If None, uses default model.
        """
        self.model = None
        self.class_names = {}
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
            
            # Try to load class indices
            class_indices_path = os.path.join(os.path.dirname(model_path), 'class_indices.json')
            if os.path.exists(class_indices_path):
                with open(class_indices_path, 'r') as f:
                    self.class_names = json.load(f)
                # Invert the dictionary to map indices to class names
                self.class_names = {v: k for k, v in self.class_names.items()}
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
            
            # Get disease name from class index
            disease_name = self.class_names.get(predicted_class, f"Disease_{predicted_class}")
            
            # Get treatment recommendation based on disease
            treatment = self.get_treatment_recommendation(disease_name)
            
            result = {
                "disease": disease_name,
                "confidence": confidence,
                "treatment": treatment
            }
            
            return result
        except Exception as e:
            print(f"Error analyzing image: {e}")
            raise
    
    def get_treatment_recommendation(self, disease_name):
        """Get treatment recommendation for a disease.
        
        Args:
            disease_name (str): Name of the disease
            
        Returns:
            str: Treatment recommendation
        """
        # Basic treatment recommendations
        if "healthy" in disease_name.lower():
            return "Plant appears healthy. Continue regular maintenance."
        elif "blight" in disease_name.lower():
            return "Remove infected leaves, improve air circulation, and consider applying fungicide."
        elif "spot" in disease_name.lower():
            return "Remove infected leaves and avoid overhead watering. Consider copper-based fungicides."
        elif "rust" in disease_name.lower():
            return "Remove infected plant material and apply appropriate fungicide. Maintain good air circulation."
        elif "mold" in disease_name.lower():
            return "Improve air circulation, reduce humidity, and consider applying fungicide."
        elif "virus" in disease_name.lower():
            return "Remove infected plants to prevent spread. Control insect vectors. No cure available."
        elif "mite" in disease_name.lower():
            return "Apply appropriate miticide. Increase humidity and remove heavily infested leaves."
        else:
            return "Please consult a plant expert for specific treatment options."