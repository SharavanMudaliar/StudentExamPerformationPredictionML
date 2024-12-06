from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction form and results
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),  # reading_score and writing_score were swapped
            writing_score=float(request.form.get('reading_score'))   # Ensure this reflects the correct input fields
        )
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])

# Route for the About page
@app.route('/about')
def about():
    return render_template('about.html')  # Render the about.html template

if __name__ == "__main__":
    app.run(host="0.0.0.0")