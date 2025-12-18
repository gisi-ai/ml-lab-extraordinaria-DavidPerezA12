# Laboratorio: Bot de Ajedrez basado en Aprendizaje Automático

## Objetivos de Aprendizaje

- Crear un modelo de Aprendizaje Automático (Machine Learning, ML) para evaluar posiciones de ajedrez basado en evaluaciones creadas por Stockfish.
- Integrar el modelo de ML en un bot de ajedrez que utilice búsqueda minimax con poda alpha-beta. 

## Descripción General

Hasta ahora, los bots que hemos implementado se basan en funciones de evaluación heurísticas diseñadas manualmente. En esta práctica, construiremos un bot que utiliza un modelo de ML para evaluar posiciones de ajedrez. El modelo se entrenará utilizando evaluaciones de posiciones generadas por Stockfish, un motor de ajedrez de código abierto ampliamente reconocido.

## Antes de empezar... 
... Copia el contenido de la práctica anterior en tu nuevo repositorio. Asegúrate de que todo funciona correctamente antes de continuar.

### Descarga del Dataset
Descarga el fichero de `evaluations.csv` en el siguiente [link](https://ceu365-my.sharepoint.com/:x:/g/personal/constantino_garciama_ceu_es/EZ6smGE56_9OgLVXbRTYdQQBw84bXtYlkUWFitJ9UlJaug?e=3ADf3z) y colócalo en la carpeta raíz de tu repositorio.

### Librerías Requeridas 

Asegúrate de usar las siguientes versiones de librerías en tu entorno de desarrollo:
- Scikit-learn 1.7.2
- Keras 3.12.0 
- Pickle con `format_version`=4 

Para comprobarlo puedes, en un terminal del sistema operativo, ejecutar:
```bash
conda list scikit-learn 
conda list keras
```
Y en una consola de python, ejecutar:
```python
import pickle
print(pickle.format_version)
```
## Guías de Entrega

### Entrega en GitHub Classroom
1. Acepta la tarea a través del enlace de GitHub Classroom
2. Clona tu repositorio asignado
3. Implementa todos los componentes requeridos
4. Incluye pruebas y documentación completas
5. Sube tu entrega final antes de la fecha límite

### Cumplimiento de API
Tu implementación será probada con scripts automatizados. La API debe seguirse exactamente - cualquier desviación resultará en fallos en las pruebas.

Se recomienda ejecutar tus propias pruebas para asegurar el cumplimiento de la API.

### Código de Honor

Está permitido discutir tareas en grupo, siempre que cada estudiante escriba sus propias soluciones desde cero y no reuse código de otros compañeros. Es una violación al código de honor compartir o consultar soluciones de otros estudiantes o de años anteriores, subir el trabajo a repositorios públicos o usar IA generativa para obtener respuestas directas o copiar soluciones.

El uso de herramientas de IA generativa, como Co-Pilot o ChatGPT, está permitido como apoyo, pero no se deben copiar soluciones completas. Usar IA para completar sustancialmente una tarea constituye una violación al código de honor.

**Las entregas sospechosas de violaciones del código de honor podrán ser sometidas a una defensa práctica. La fecha y hora de la defensa será decidida por el profesor y no será negociable.**

**Las violaciones graves del código de honor resultarán en la pérdida de una o varias convocatorias así como la apertura de expediente académico.**


# Dataset y Tareas

El fichero `evaluations.csv` contiene 1_680_001 evaluaciones de posiciones de ajedrez realizadas por Stockfish. Cada fila contiene una representación FEN (Forsyth-Edwards Notation) de una posición, una notación estándar que describe el tablero, el turno, derechos de enroque y otras propiedades relevantes. Adicionalmente, cada FEN viene acompañada de las siguientes columnas que describen la evaluación de Stockfish para esa posición:
* `is_mate`: Un valor booleano que indica si la posición es un mate forzado (se puede dar mate en un número finito de movimientos).
* `cp`: La evaluación de la posición correspondiente en centipawns (céntimos de peón, cp). Evaluaciones muy positivas indican una ventaja para las blancas, mientras que evaluaciones muy negativas indican una ventaja para las negras. El rango de valores es típicamente entre -20000 y +20000. Este valor solo está presente si `is_mate` es `False`.
* `mate_distance`: El número de movimientos hasta el mate (positivo si las blancas dan mate, negativo si las negras dan mate). Por ejemplo, -3 indica que las negras pueden dar mate en 3 movimientos. Esta columna solo está presente si `is_mate` es `True`.

El fichero tiene el siguiente formato (las primeras filas son solo un ejemplo):
```
fen,cp,is_mate,mate_distance
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -,19.0,False,
8/8/2N2k2/8/1p2p3/p7/K7/8 b - -,0.0,False,
8/1r6/2R2pk1/6pp/3P4/6P1/5K1P/8 w - -,0.0,False,
8/5kp1/6N1/4K3/4N3/8/8/8 w - -,,True,23.0
R4nk1/6p1/8/5pb1/P1pN2b1/7p/2B3P1/5K2 w - -,320.0,False,
...
```

Nótese que este conjunto de datos, aunque aparentemente grande, es solo una pequeña fracción del espacio total de posiciones de ajedrez posibles. Esto implica que el modelo de ML que entrenes puede tener problemas para generalizar a posiciones no vistas durante el entrenamiento (ver sección final de `ideas para la práctica voluntaria`). A pesar de esta limitación, **NO está permitido usar otras fuentes de datos para esta práctica**. Sí está permitido, si lo deseas, usar técnicas de muestreo de datos o de *augmentation*.

**Tu tarea es utilizar este conjunto de datos para entrenar un modelo de ML que pueda evaluar posiciones de ajedrez.** Concretamente, debes completar las siguientes dos tareas. Fíjate que los enunciados son intencionalmente abiertos para permitirte tomar decisiones de diseño importantes. **Es fundamental que documentes y justifiques todas las decisiones que tomes durante la implementación.**




## Tarea 1: Creación de un modelo de ML para evaluar posiciones de ajedrez

**Esta tarea debe ser implementada en un Jupyter Notebook llamado `ml_model.ipynb`.** No uses rutas absolutas en el notebook; asume que el fichero `evaluations.csv` está en la misma carpeta que el notebook.

Antes de empezar esta tarea, se recomienda leer también la tarea 2, ya que el modelo que crees aquí será utilizado en la implementación del bot de ajedrez, y este tiene ciertos requisitos.

Este notebook debe contener, necesariamente, las siguientes secciones:

### **1. Análisis Exploratorio de Datos (EDA)** 
Analiza la distribución de las evaluaciones en el dataset:
* Proporción de mates vs. evaluaciones cp
* Distribución de valores cp (histograma)
* Distribución de mate_distance

Este análisis te ayudará a tomar decisiones sobre normalización, balanceo de clases y arquitectura del modelo.

### **2. Carga y Preprocesamiento de Datos**
Carga el fichero `evaluations.csv` y realiza cualquier preprocesamiento necesario. Como mínimo, hay tres aspectos que requieren tu atención:
- Debes convertir las posiciones FEN a características numéricas adecuadas para el modelo de ML. Posibles ideas para transformar el tablero a características numéricas son 1) el uso de características ya empleadas en las heurísticas diseñadas manualmente, como la presencia de piezas en ciertas casillas, el conteo de piezas, etc.; 2) El uso del concepto de [bitboard](https://www.chessprogramming.org/Bitboards). ¡No olvides añadir el turno como característica!

Puedes apoyarte en python-chess para cargar el tablero en un objeto:
```python 
board = chess.Board(fen)
```

* Debes manejar las dos métricas de evaluación de Stockfish (`cp` y `mate_distance`) de manera adecuada. Una posible estrategia es entrenar dos modelos separados, uno para cada métrica. Otra posible estrategia es transformar las evaluaciones de mate en valores numéricos continuos que puedan ser manejados por un único modelo.
* El rango de valores de las evaluaciones es muy amplio (ver apartado sobre EDA), lo que puede dificultar el entrenamiento del modelo, especialmente para rangos con poca representación. Considera cómo gestionar este problema. Posibles ideas son normalizar las evaluaciones, truncar valores, agrupar evaluaciones en subrangos, etc.  

Explica claramente las decisiones que tomes en esta sección.

### **3. Selección y Entrenamiento del Modelo** 
Elige un modelo de ML adecuado para la tarea de regresión. Entrena el modelo utilizando el conjunto de datos preprocesado. Si lo deseas, puedes comparar varios modelos antes de elegir el mejor. El resultado final debe de la sección debe ser, o bien:
- Un fichero `ml_model.pkl` que contenga el modelo entrenado serializado, guardado utilizando la librería `pickle`.
- Un fichero que `ml_model.keras` que contenga los pesos del modelo entrenado, en caso de que utilices Keras.

Puedes guardar y cargar el modelo utilizando el siguiente código de ejemplo:

```python
# Modelos de Scikit-learn
import pickle

# Guardar el modelo 
with open('ml_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Cargar el modelo
with open('ml_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Modelos de Keras

# Guardar el modelo
model.save('ml_model.keras')

# Cargar el modelo
model = keras.models.load_model('ml_model.keras')
```

El anterior fichero debe ser añadido al repositorio para ser utilizado en la Tarea 2. Si el fichero es muy grande, es posible que necesites usar [Git LFS](https://git-lfs.github.com/). 

Explica claramente las decisiones que tomes en esta sección.

### 4. Evaluación del rendimiento del modelo
Evalúa el rendimiento del modelo utilizando métricas adecuadas, como el error cuadrático medio (MSE) o el error absoluto medio (MAE). Una idea interesante a tener en cuenta es que errores en posiciones igualadas (por ejemplo, |cp| < 100) son más críticos que errores en ventajas decisivas, por lo que podrías considerar ponderar los errores en función de cómo de decidida esté la partida.

Incluye gráficos que muestren la distribución de los errores y cualquier otra visualización que consideres relevante. Indica claramente cuáles son tus métricas finales de rendimiento recogiéndolas en una tabla.

Explica claramente las decisiones que tomes en esta sección.

## Tarea 2: Implementación de un bot de ajedrez utilizando el modelo de ML 
Esta tarea debe ser implementada en el fichero `bots/ml_bot.py`. El bot debe utilizar el modelo de ML entrenado en la Tarea 1 para evaluar posiciones de ajedrez durante la búsqueda minimax con poda alpha-beta. Para esto, procederemos en dos pasos:

### Tarea 2.1: Crea un Wrapper para el modelo de ML

Implementa `MLModel` en `bots/ml_bot.py`. Su constructor debe recibir un argumento `model_file` que le indique el fichero desde el que debe cargar el modelo de ML. 
`MLModel` deberá implementar un método `predict` que reciba un `ChessGameState` y devuelva su predicción para el tablero en centipawns (es decir, en la escala original con valores entre -20000 y 20000). Concretamente:

```python
class MLModel:
    def __init__(
        self,
        model_file: str,
        # You may add other parameters if needed, but always with default values
    ):
        pass

    def predict(self, state: ChessGameState) -> float:
        pass
```
 
### Tarea 2.2: MLBot
Implementa `MLBot` en `bots/ml_bot.py`. Su constructor debe recibir un argumento `model_file` que le indique el fichero desde el que debe cargar el modelo de ML. Internamente, el bot usará este fichero para crear una instancia de `MLModel` que almacenará en `self.ml_model`. Finalmente, el bot debe implementar un método `evaluate_position`.  Concretamente:

```python
class MLBot(ChessBot):
    def __init__(
        self,
        model_file: str,
        max_depth: int = 3,
        # You may add other parameters if needed, but always with default values
    ):
        # MLBot MUST HAVE A ML_MODEL ATTRIBUTE
        self.ml_model = ... 

    def evaluate_position(self, state: ChessGameState, player: chess.Color) -> float:
        pass

    # Other methods as required by the ChessBot API
```
 
 
Es importante que, antes de realizar la tarea 1, reflexiones sobre los requerimientos de las funciones de evaluación en una búsqueda minimax, ya que esto puede influir en el diseño del modelo de ML:
* Las evaluaciones deben ser rápidas de calcular, ya que se llamarán muchas veces durante la búsqueda. Como guía aproximada, tu función `evaluate_position` debe ser capaz de evaluar entre ~1000 posiciones por segundo en un portátil estándar (sin GPU). Como límite inferior, tu función debe ser capaz de evaluar al menos ~500 posiciones por segundo.
* En nuestra implementación de `ChessProblem`, nuestra función de utilidad para estados terminales toma valores de 0, 0.5 o 1. Esto implica que las evaluaciones del modelo de ML (que obtendrás con `MLModel.predict()`) deben ser escaladas a este rango. Adicionalmente, fíjate que `evaluate_position` recibe como argumento el color del jugador. Por ejemplo, si tu modelo de ML devuelve evaluaciones entre -20000 y 20000, y una posición es muy favorable para negras con una puntuación de -18000:
    * `evaluate_position(board, chess.BLACK)` debería devolver 0.95.
    * `evaluate_position(board, chess.WHITE)` debería devolver 0.05.

Si crees necesario documentar alguna decisión de diseño adicional, puedes añadir un fichero `ml_bot_design.md` al repositorio.


## Ideas para la Práctica Voluntaria Final (No Evaluable Ahora)
Por motivos didácticos y de cómputo, el tamaño del conjunto de datos es limitado en comparación con el espacio total de posiciones de ajedrez posibles. Esto puede llevar a que el modelo de ML tenga dificultades para generalizar a posiciones no vistas durante el entrenamiento. Cuando pruebes el bot en partidas reales, es posible que notes que el bot juega de forma brillante en algunas posiciones, pero comete errores evidentes en otras. Uno de los temas a los que puedes dedicar la última práctica (voluntaria) del curso es a mejorar la capacidad de generalización del modelo o, al menos, a detectar cuando sus predicciones son fiables. Algunas ideas que podrías explorar incluyen: 
* Entrenar el modelo empleando más datos. Puedes emplear: https://database.lichess.org/ 
* Usar técnicas de *uncertainty estimation* para detectar cuándo el modelo no es fiable. La idea básica es que el modelo de ML predecirá no solo una evaluación, sino también una medida de incertidumbre asociada a esa predicción. Si la incertidumbre es alta, el bot podría optar por usar una función de evaluación heurística en lugar del modelo de ML. Otra opción en esta línea (más sencilla) es emplear alguna función heurística simple para contrastar con la predicción del modelo de ML. Si difieren mucho, el bot podría optar por usar la función heurística.

