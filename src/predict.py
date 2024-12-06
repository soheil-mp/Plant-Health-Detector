import argparse
import os
from plant_detector import PlantDiseaseDetector

def predict_image(args):
    """Make predictions on plant images.
    
    Args:
        args: Parsed command line arguments
    """
    # Initialize detector
    detector = PlantDiseaseDetector(model_path=args.model_path)
    
    if args.image:
        # Single image prediction
        result = detector.analyze_image(args.image)
        print("\nAnalysis Results:")
        print(f"Disease: {result['disease']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Treatment: {result['treatment']}")
    
    elif args.input_dir:
        # Batch processing
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
        for filename in os.listdir(args.input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(args.input_dir, filename)
                result = detector.analyze_image(image_path)
                
                print(f"\nAnalyzing {filename}:")
                print(f"Disease: {result['disease']}")
                print(f"Confidence: {result['confidence']:.2f}")
                print(f"Treatment: {result['treatment']}")

def main():
    parser = argparse.ArgumentParser(description='Predict plant diseases from images')
    parser.add_argument('--model_path', required=True, help='Path to trained model (.keras file)')
    parser.add_argument('--image', help='Path to single image for prediction')
    parser.add_argument('--input_dir', help='Directory containing images for batch prediction')
    parser.add_argument('--output_dir', default='results', help='Directory to save results')
    
    args = parser.parse_args()
    
    if not args.image and not args.input_dir:
        parser.error("Either --image or --input_dir must be provided")
    
    predict_image(args)

if __name__ == '__main__':
    main() 