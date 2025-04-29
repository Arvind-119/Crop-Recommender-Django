# Crop Recommendation System

This is a Crop Recommendation System with Django as User Interface.

## Requirements
- Python 3.8+
- Django 4.2
- NumPy
- Pandas
- scikit-learn

## Setup Instructions

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train the machine learning model:
   ```
   python train.py
   ```
   This will create the required model files in the `models/` directory.

3. Run the Django development server:
   ```
   python manage.py runserver
   ```

4. Access the application in your browser at `http://127.0.0.1:8000/`

## About the Project

This Django application recommends suitable crops based on soil and environmental parameters:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- pH
- Rainfall

The system uses Django web framework as the user interface, providing an intuitive way for users to input soil parameters and receive crop recommendations.

## File Structure

- `crop_project/` - Django project settings
- `crop_app/` - Main application code
  - `views.py` - Contains the ML logic and view functions
  - `urls.py` - URL routing
- `models/` - Contains ML model training script and saved model files
- `templates/crop_app/` - HTML templates
- `static/` - Static files (CSS, images)

## Model Information

The model is a Random Forest Classifier trained on crop data. It predicts the best crop for the given soil and environmental conditions. The model files are:

- `models/model.pkl` - The trained machine learning model
- `models/minmax_scaler.pkl` - Min-Max scaler for input normalization
- `models/standard_scaler.pkl` - Standard scaler for feature scaling 