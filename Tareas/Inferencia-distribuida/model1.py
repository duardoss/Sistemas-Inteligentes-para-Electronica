################################################################################
# model1 - LLM (Gemma) servido localmente con llama.cpp

from openai import OpenAI

# Definición del cliente local (llama.cpp)
client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="sk-no-key-required"
)

def model1(R, G, B):
    # Implementación del modelo 1
    TEXTO = (
            f"Tengo lecturas tomadas directamente de un sensor TCS3200: R={R}, G={G}, B={B}. "
            f"En este sensor, UN VALOR MÁS BAJO SIGNIFICA MAYOR INTENSIDAD del color correspondiente. "
            f"Analiza cuál de los tres componentes (R, G o B) tiene el valor numérico MÁS BAJO. "
            f"Responde ÚNICAMENTE con una de estas tres palabras exactas: 'Rojo', 'Verde' o 'Azul'. "
            f"No agregues explicaciones ni puntuación adicional."
        )

    clase = "Clase del modelo 1"  # Valor por defecto en caso de error

    try:
        completion = client.chat.completions.create(
            model="LLaMA_CPP",
            messages=[
                {
                    "role": "system",
                    "content": "Eres un asistente útil que responde exactamente con la clase solicitada sin texto adicional."
                },
                {
                    "role": "user",
                    "content": TEXTO
                }
            ],
            timeout=10.0  # Evita bloqueos por demoras prolongadas
        )

        # Extraer la respuesta del modelo
        respuesta = completion.choices[0].message.content
        if respuesta:
            # Elimina <end_of_turn> y limpia espacios o saltos de línea sobrantes
            clase = respuesta.replace("<end_of_turn>", "").strip()

    except Exception as e:
        print(f"Error al comunicarse con el modelo en el puerto 8080: {e}")

    return clase