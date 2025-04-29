from django.shortcuts import render
import numpy as np
import pandas as pd
import pickle
import os
from django.conf import settings

# Define crop dictionary for display
crop_dict_display = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Load the trained model and scalers from Django models directory
model_path = os.path.join(settings.BASE_DIR, 'models', 'model.pkl')
minmax_path = os.path.join(settings.BASE_DIR, 'models', 'minmax_scaler.pkl')
scaler_path = os.path.join(settings.BASE_DIR, 'models', 'standard_scaler.pkl')

# Initialize model and scalers
model = None
mx = None
sc = None

# Load model and scalers if they exist
if os.path.exists(model_path) and os.path.exists(minmax_path) and os.path.exists(scaler_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    with open(minmax_path, 'rb') as f:
        mx = pickle.load(f)
        
    with open(scaler_path, 'rb') as f:
        sc = pickle.load(f)
    
    print("Model and scalers loaded successfully!")
else:
    print("Model or scaler files not found! Please run the training script in the models directory first.")

def index(request):
    return render(request, 'crop_app/index.html')

def predict(request):
    result = ""
    if request.method == 'POST':
        try:
            # Check if model is loaded
            if model is None or mx is None or sc is None:
                return render(request, 'crop_app/index.html', 
                            {'result': "Error: Model not loaded. Please run the training script in the models directory first."})
            
            # Get input values
            N = float(request.POST.get('Nitrogen'))
            P = float(request.POST.get('Phosporus'))
            K = float(request.POST.get('Potassium'))
            temp = float(request.POST.get('Temperature'))
            humidity = float(request.POST.get('Humidity'))
            ph = float(request.POST.get('pH'))
            rainfall = float(request.POST.get('Rainfall'))

            # Create feature array
            feature_list = [N, P, K, temp, humidity, ph, rainfall]
            single_pred = np.array(feature_list).reshape(1, -1)
            
            # Apply the same transformations as training data
            mx_features = mx.transform(single_pred)
            sc_mx_features = sc.transform(mx_features)
            
            # Make prediction
            prediction = model.predict(sc_mx_features)
            
            # Display result
            if prediction[0] in crop_dict_display:
                crop = crop_dict_display[prediction[0]]
                result = f"{crop} is the best crop to be cultivated right there"
            else:
                result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
        except Exception as e:
            result = f"Error: {str(e)}"
    
    return render(request, 'crop_app/index.html', {'result': result}) 