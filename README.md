# 🌸 Iris Flower Classifier (Flask + ANN)

This project is a **Flask web application** that predicts the species of an Iris flower 
(`setosa`, `versicolor`, or `virginica`) based on four input measurements:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The app uses a **trained Artificial Neural Network (ANN)** built with TensorFlow/Keras, 
and a pre-fitted scaler (via joblib) to preprocess the input data.
---
## 🚀 Features
- Web form built with **Flask-WTF** and **WTForms** for secure input handling.
- CSRF protection enabled via Flask’s `SECRET_KEY`.
- Model predictions served through a clean Flask route.
- Templates (`home.html` and `prediction.html`) for user interaction and results display.

---

## 📂 Project Structure
```
Iris_Dataset_ANN/
├── app.py                # Main Flask application
├── final_iris.h5         # Trained Keras model
├── scaler_iris.pkl       # Pre-fitted scaler
├── templates/
│   ├── home.html         # Input form template
│   └── prediction.html   # Results template
└── README.md             # Project description
```

---

## ⚙️ Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/abhi12Abhiuo1208/Iris_Dataset_ANN.git
   cd Iris_Dataset_ANN
   ```
**Note:** After cloning the repository please go to `model.ipynb` and run the cells to generate the model `.h5` and scaled object joblib `.pkl` file.  

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   python app.py
   ```

4. Open in your browser:
   ```
   http://127.0.0.1:5000/
   ```

---

## 🧪 Example Input
- Sepal Length: `5.1`  
- Sepal Width: `3.5`  
- Petal Length: `1.4`  
- Petal Width: `0.2`  

**Prediction:** `setosa`

---

## 📖 Dataset Reference
This project is based on the classic [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris), 
a well-known dataset in machine learning for classification tasks.

---

## 👨‍💻 Author
Developed by **Abhishek**  
Connect with me on [Linkedin](https://www.linkedin.com/in/abhiuo007/)  
📍 Bengaluru, India
```

---

