import tensorflow as tf
import numpy as np

libras = np.array([100, 170, 38, 212, 315, 122, 338], dtype=float)
kilos = np.array([45.3592, 77.1107, 17.2365, 96.1616, 142.8816, 55.3383, 153.3142], dtype = float)

oculta1 = tf.keras.layers.Dense(units=3, input_shape = [1])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2, salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss="mean_squared_error"
)

print("Comenzando entrenamiento...")
historial = modelo.fit(libras, kilos, epochs=70, verbose=False)
print("Modelo entrenado!")

import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])

print("Hagamos una predicción de 50 libras a ver si funciona...")
resultado = modelo.predict([50.0])
print("El resultado es "+str(resultado)+" kilos, y debería ser 22,6796")
