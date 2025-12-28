# -------------------------------
# Imports
# -------------------------------
from flask import Flask, render_template, session, url_for, redirect
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField


# -------------------------------
# Helper function for prediction
# -------------------------------
def return_prediction(model, scaler, sample_json):
    """
    Takes in a dictionary of flower measurements,
    scales the values, and returns the predicted species.
    """

    # Extract values from dictionary
    s_len = sample_json["sepal_length"]
    s_wid = sample_json["sepal_width"]
    p_len = sample_json["petal_length"]
    p_wid = sample_json["petal_width"]

    # Arrange into 2D array for model input
    flower = [[s_len, s_wid, p_len, p_wid]]

    # Scale the input using the pre-fitted scaler
    flower = scaler.transform(flower)

    # Define class labels (must match training order)
    classes = np.array(['setosa', 'versicolor', 'virginica'])

    # Model prediction returns probability distribution
    class_ind = model.predict(flower)[0]

    # Return the class with the highest probability
    return classes[np.argmax(class_ind)]


# -------------------------------
# Flask App Setup
# -------------------------------
app = Flask(__name__)

# Secret key required for CSRF protection and session management
# In production, load this from environment variables instead of hardcoding
app.config['SECRET_KEY'] = 'mysecretkey'


# -------------------------------
# Form Definition
# -------------------------------
class FlowerForm(FlaskForm):
    """
    Defines the input form for flower measurements.
    """
    sepal_length = StringField('Sepal Length')
    sepal_width = StringField('Sepal Width')
    petal_length = StringField('Petal Length')
    petal_width = StringField('Petal Width')
    submit = SubmitField('Predict')


# -------------------------------
# Routes
# -------------------------------
@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Home route: displays the input form.
    On POST, saves form data into session and redirects to prediction page.
    """
    form = FlowerForm()

    if form.validate_on_submit():
        # Store user inputs in session
        session['sepal_length'] = form.sepal_length.data
        session['sepal_width'] = form.sepal_width.data
        session['petal_length'] = form.petal_length.data
        session['petal_width'] = form.petal_width.data

        return redirect(url_for('prediction'))

    return render_template('home.html', form=form)


# -------------------------------
# Model and Scaler Loading
# -------------------------------
flower_model = load_model('final_iris.h5')
scaler = joblib.load('scaler_iris.pkl')


@app.route('/prediction')
def prediction():
    """
    Prediction route: retrieves input values from session,
    runs them through the model, and renders the result.
    """
    # Build dictionary from session values
    content = {
        'sepal_length': float(session['sepal_length']),
        'sepal_width': float(session['sepal_width']),
        'petal_length': float(session['petal_length']),
        'petal_width': float(session['petal_width'])
    }

    # Get prediction
    results = return_prediction(flower_model, scaler, content)

    # Pass both inputs and prediction to template
    return render_template(
        'prediction.html',
        prediction=results,
        sepal_length=content['sepal_length'],
        sepal_width=content['sepal_width'],
        petal_length=content['petal_length'],
        petal_width=content['petal_width']
    )


# -------------------------------
# Run the App
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)