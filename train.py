import os
import sys

def main():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the current directory to the Python path
    sys.path.append(current_dir)
    
    # Import and run the training script from models directory
    try:
        # Import the module before execution to access its variables later
        import models.train_model as tm
        # The accuracy variable should be available after import
        accuracy = tm.accuracy
        print(f"Training completed successfully!")
        print(f"Model accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    except Exception as e:
        print(f"Error during training: {str(e)}")

if __name__ == "__main__":
    main() 