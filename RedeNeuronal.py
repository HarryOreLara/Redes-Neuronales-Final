import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split, cross_val_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
import pickle
from datos import data


# Paso 1: Recopila datos etiquetados (frases y estados de ánimo)

df = pd.DataFrame(data)


# Paso 2: Preprocesamiento de texto
def preprocess_text(text):
    # Conversión a minúsculas
    text = text.lower()

    # Eliminación de caracteres especiales y puntuación
    text = re.sub(r'[^\w\s]', '', text)

    return text

df['frase'] = df['frase'].apply(preprocess_text)





# Paso 3: Tokenización y secuenciación de texto
max_words = 10000  # Número máximo de palabras a considerar
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['frase'])
X = tokenizer.texts_to_sequences(df['frase'])


# Paso 4: Padding de secuencias
maxlen = 30  # Longitud máxima de las secuencias
X = pad_sequences(X, maxlen=maxlen, padding='pre')





# Paso 5: División de datos en entrenamiento y prueba
y = df['estado_animo']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convierte las etiquetas categóricas a números enteros
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Convierte las etiquetas categóricas a etiquetas de clase única
#y_train_encoded = tf.keras.utils.to_categorical(y_train_encoded, num_classes=3)
#y_test_encoded = tf.keras.utils.to_categorical(y_test_encoded, num_classes=3)
num_classes = len(label_encoder.classes_)
y_train_encoded = tf.one_hot(y_train_encoded, depth=num_classes)
y_test_encoded = tf.one_hot(y_test_encoded, depth=num_classes)



model = keras.Sequential([
    Embedding(input_dim=max_words, output_dim=256, input_length=maxlen),
    Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(9, activation='softmax')  # Cambia el número de clases de salida a 3
])




# Paso 7: Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'],
              loss_weights=[0.2, 0.2, 0.6])


model.summary()

# Paso 8: Entrenamiento del modelo
model.fit(X_train, y_train_encoded, epochs=5000, batch_size=1, validation_split=0.2)


model.save('redes_neuronales.h5')
# Guarda el tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
    
    # Guarda el codificador de etiquetas
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)




# Paso 9: Evaluación del modelo
# loss, accuracy = model.evaluate(X_test, y_test_encoded)
# print(f"Exactitud (Accuracy): {accuracy}")

loss, accuracy = model.evaluate(X_test, y_test_encoded)

print("Pérdida en datos de prueba:", loss)
print("Precisión en datos de prueba:", accuracy)



# Paso 10: Predicción de estados de ánimo
nueva_frase = ["Me siento solo"]
nueva_frase = [preprocess_text(frase) for frase in nueva_frase]
nueva_frase_sequence = tokenizer.texts_to_sequences(nueva_frase)
nueva_frase_sequence = pad_sequences(nueva_frase_sequence, maxlen=maxlen)  # Asegúrate de que tenga la misma longitud máxima
prediccion = model.predict(nueva_frase_sequence)
prediccion_clase = np.argmax(prediccion, axis=-1)

# Mapea la clase predicha nuevamente a la etiqueta original
clase_predicha = label_encoder.inverse_transform(prediccion_clase)
print(f"Frase: {nueva_frase[0]}")
print(f"Estado de ánimo predicho: {clase_predicha[0]}")

