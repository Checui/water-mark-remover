# water-mark-remover
Trabajo mg
link a dataset: https://www.kaggle.com/code/therealcyberlord/watermark-removal-using-convolutional-autoencoder

## Resumen
Objetivo: Comparar algoritmos quitar marca de agua.

- Pre-entrenamiento 1: pasar de marca de agua a marca de agua.
- Pre-entrenamiento 2: original a original
- Pre-entrenamiento 3: Imagen con patches a imagen original.
- Pre-entrenamiento 4: entrenar función de recompensa (0/1)

Optimización de hiperparámetros:
- Learning rate
- Número de bloques
-----

- U-Net + discriminador

- U-Net + función de recompensa.

-----

Nota: la función de recompensa sigue la misma arquitectura que el discriminador.