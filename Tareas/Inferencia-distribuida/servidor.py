################################################################################
# Variables

_LISTA_LLAVES = ["MAC", "R", "G", "B"]

_PATH_DATA = "C:\\Users\\esteb\\Downloads\\data"

_FILENAME= "info.txt"

_VERBOSE = True

################################################################################

from datetime import datetime
from flask import Flask, request, jsonify
import json
import os

################################################################################
# Selección del modelo (DEJAR SOLO UNA LÍNEA SIN COMENTAR, SI NO, el programa usa el último modelo)

from model1 import model1 as predecir   # Gemma
#from model2 import model2 as predecir  # Árbol de decisión
#from model3 import model3 as predecir  # Perceptrón multiclase

################################################################################
# main 

app = Flask(__name__)

# Crear carpeta donde se guardará la información
# No generar error en caso que la ruta ya exista
try:
    os.mkdir(_PATH_DATA)
except:
    pass

@app.route('/data', methods=['POST'])
def receive_data():
    try:

        # Obtener datos
        data = request.get_json(force=True)

        # En caso que no estén todas las keys, da error
        if not all(key in data for key in _LISTA_LLAVES):

            _VERBOSE and print("JSON inválido recibido", flush=True)

            # Retornar
            return jsonify(
                {
                    "status": "error",
                    "message": f"Formato inválido, se esperan las siguientes keys: {_LISTA_LLAVES}"
                }), 400

        # Código en caso que no se cumpla condición anterior

        _VERBOSE and print(f"Datos recibidos: {data}", flush=True)

        # Obtener la hora actual
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")

        # Guardar en archivo (una línea JSON por registro)
        _ = f"{_PATH_DATA}/{_FILENAME}"
        with open(_, "a", encoding="utf-8") as f:
            f.write(f"{timestamp},{json.dumps(data)}\n")

        # Obtener el valor de RGB
        R = data["R"]
        G = data["G"]
        B = data["B"]

        # Predecir con el modelo elegido arriba
        print(f"Prediciendo con el modelo para R={R}, G={G}, B={B}")
        clase = predecir(R, G, B)
        print(f"Predicción obtenida: {clase}")

        # Retornar
        return jsonify(
            {
                "status": "ok",
                "data": clase
            }), 200

    # Exception
    except Exception as e:

        # Retornar
        return jsonify(
            {
                "status": "error",
                "message": str(e)
            }), 400

################################################################################
# __main__
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
