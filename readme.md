# API de Predicción con XGBoost

Esta es una API basada en **FastAPI** que utiliza un modelo de **XGBoost** entrenado para hacer predicciones sobre un conjunto de datos. La API recibe las características de una muestra y devuelve la predicción correspondiente.

## Ejecucion del proyecto

### Inicializa un entorno virtual 

```bash
python -m venv venv
```

**Windows**
```bash
./venv/Scripts/activate
```

**MAC**
```bash
source venv/bin/activate
```


### Instalación de las Dependencias

Ejecuta el siguiente comando para instalar todas las dependencias listadas en el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Entrenamiento del modelo 

Entrenar el modelo con el siguiente comando:

```bash
python train_model.py
```

Lo cual generara en la carpeta **models/** el .pkl del modelo entrenado

## Levantar la API 

Levantar la API con el siguiente comando:

```bash
python app.py
```

Lo cual levantara el servicio localmente, se puede revisar si esta corriendo por medio de el [health_status](http://127.0.0.1:1234/health) endpoint

## Pruebas Chi2 y Kolmogorov-Smirnoff

Correr el siguiente script

```bash
python distributions.py
```
El cual debe regresar los valores p de cada una de las variables clasificadas por categoricas y continuas


## Demo a la API

Correr el siguiente script

```bash
python demo.py
```

El cual debe regresar una prediccion de acuerdo a los datos que contenga la variable **data**


## Generar CSV con valores Y

Correr el siguiente script

```bash
python script.py
```

El cual debe regresar una prediccion de acuerdo a los datos que contenga la variable **data**



