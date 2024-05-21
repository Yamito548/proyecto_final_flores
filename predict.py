import pickle
from flask import Flask, request, jsonify
import numpy as np

# Cargar el modelo de machine learning
with open('modelo_diabetes.pkl', 'rb') as f:
    modelo = pickle.load(f)

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def predecir_diabetes():
    # Obtener los datos del cuerpo de la solicitud POST
    data = request.get_json(force=True)

    # Convertir los datos en un array de numpy
    valores = np.array(list(data.values()))

    # Realizar la predicción con el modelo
    prediccion = modelo.predict(valores.reshape(1, -1))

    # Devolver la predicción como respuesta
    return jsonify(prediccion.tolist())

if __name__ == '__main__':
    app.run(debug=True)
