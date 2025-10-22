from flask import Flask, request, render_template
import pickle
import pandas as pd

# Initialize Flask
app = Flask(__name__)

# Load your trained model
model = pickle.load(open(r"C:\Users\ADMIN\note\Decisiontree\FitOrNotFit\model.pkl", "rb"))

# Home route
@app.route('/', methods=["POST", "GET"])
def predict():
    result = None
    if request.method == 'POST':
        # Get user input from form
        age = float(request.form.get("age"))
        height = float(request.form.get("height"))
        weight = float(request.form.get("weight"))
        heart_rate = float(request.form.get("heart_rate"))
        blood_pressure = float(request.form.get("blood_pressure"))
        sleep_hours = float(request.form.get("sleep_hours"))
        nutrition = float(request.form.get("nutrition_quality"))
        activity_index = float(request.form.get("activity_index"))
        
        # Encode categorical inputs
        smokes_input = request.form.get("smokes").lower()
        smokes = 1 if smokes_input in ['1','yes','y'] else 0
        
        gender_input = request.form.get("gender").lower()
        gender = 1 if gender_input in ['1','m','male'] else 0

        # Create DataFrame for model
        new_data = pd.DataFrame([{
            'age': age,
            'height_cm': height,
            'weight_kg': weight,
            'heart_rate': heart_rate,
            'blood_pressure': blood_pressure,
            'sleep_hours': sleep_hours,
            'nutrition_quality': nutrition,
            'activity_index': activity_index,
            'smokes': smokes,
            'gender': gender
        }])

        # Predict
        result = model.predict(new_data)[0]

    return render_template("Home.html", result=result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
