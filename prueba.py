import os
import keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle



#####################################
#####################################
#####################################
#####################################

#######ACA SE REALIZARON  5K DE ENTRENAMIENTOS PARA LAS REDES NEURONALES
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################

# Carga el modelo
model = keras.models.load_model('redes_neuronales.h5')

# Carga el codificador de etiquetas
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
# Función de preprocesamiento de texto
def preprocess_text(text):
    # Conversión a minúsculas
    text = text.lower()

    # Eliminación de caracteres especiales y puntuación
    text = re.sub(r'[^\w\s]', '', text)

    return text

# Prepara la frase
frase = "Me siento solo todos los dias"
frase = preprocess_text(frase)
maxlen = 30
frase_sequence = tokenizer.texts_to_sequences([frase])
frase_sequence = pad_sequences(frase_sequence, maxlen=maxlen)

# Predice los sentimientos
prediccion = model.predict(frase_sequence)
prediccion_clase = np.argmax(prediccion, axis=-1)

# Interpreta los resultados
clase_predicha = label_encoder.inverse_transform(prediccion_clase)

# Imprime los resultados
print(f"Frase: {frase}")
print(f"Sentimiento predicho: {clase_predicha[0]}")
