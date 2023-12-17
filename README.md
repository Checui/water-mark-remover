# water-mark-remover
link a dataset: https://www.kaggle.com/code/therealcyberlord/watermark-removal-using-convolutional-autoencoder

La versión final del notebook [entrega.ipynb](entrega.ipynb) se encuentra subido a colab puesto que tiene un tamaño superior a 100Mb:
https://colab.research.google.com/drive/154RNtIqs_P5dqkchrAp-k0a9LLx3YL9s?usp=sharing

## Resumen
Objetivo: Comparar diseños basados en U-Net para eliminar marcas de agua.

- Pre-entrenamiento 1: pasar de marca de agua a marca de agua.
- Pre-entrenamiento 2: original a original
- Pre-entrenamiento 3: Imagen con patches a imagen original.
- Pre-entrenamiento 4: entrenar función de recompensa (0/1)

Optimización de hiperparámetros:
- Learning rate
-----

- U-Net + discriminador

- U-Net + función de recompensa.

-----

Nota: la función de recompensa sigue la misma arquitectura que el discriminador.

## Cómo usar Poetry

**Paso 1:** Instalar Poetry
```bash
pip install poetry
```

**Paso 2:** Activar el entorno virtual
```bash
poetry shell
```

**Paso 3:** Instalar las dependencias
```bash
poetry install
```

**Paso 4:** Instalar el paquete local
```bash
pip install . -e
```
La bandera -e permite modificar el paquete sin tener que volver a instalarlo.

Para añadir dependencias:
```bash
poetry add nombre-del-paquete
```
