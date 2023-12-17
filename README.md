# water-mark-remover
link a dataset: https://www.kaggle.com/code/therealcyberlord/watermark-removal-using-convolutional-autoencoder

La versión final del notebook [entrega.ipynb](entrega.ipynb) se encuentra subido a colab puesto que tiene un tamaño superior a 100Mb:
https://colab.research.google.com/drive/154RNtIqs_P5dqkchrAp-k0a9LLx3YL9s?usp=sharing

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
