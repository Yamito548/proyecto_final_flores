from flask import Flask, request, render_template_string, Response
import joblib
import numpy as np

# Cargar el modelo entrenado
model = joblib.load('modelo_diabetes.pkl')

# Inicializar Flask
app = Flask(__name__)

# Ruta para mostrar el formulario HTML
@app.route('/')
def form():
    # Abrir el archivo index.html y leer su contenido con encoding UTF-8
    with open('index.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    return Response(html_content, content_type='text/html; charset=utf-8')

# Ruta para manejar la predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos del formulario
    data = request.form
    input_data = np.array([
        data['age'], data['hypertension'], data['heart_disease'],
        data['smoking_history'], data['bmi'], data['HbA1c_level'],
        data['blood_glucose_level']
    ], dtype=float).reshape(1, -1)
    
    # Realizar la predicción
    prediction = model.predict(input_data)
    
    # Preparar la respuesta
    result = "positivo" if prediction[0] == 1 else "negativo"
    return f"<h2>La predicción es: {result}</h2>"

# Ejecutar la aplicación Flask
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
