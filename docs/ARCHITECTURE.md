# Arquitectura del sistema de neurofeedback EEG

## 1. Objetivo y diferencia frente a imaginación motora

El sistema recibirá continuamente el EEG del Unicorn y producirá un índice
numérico de atención actualizado cinco veces por segundo. En imaginación motora
el objetivo era decidir entre dos clases discretas, como izquierda y derecha.
Aquí la salida es continua: por ejemplo, un valor interno normalizado y una
representación visual limitada entre 0 y 100.

La razón Theta/Beta es una característica candidata, no una lectura directa e
infalible de la atención. Su valor depende del sujeto, los canales, las bandas,
la referencia, los artefactos y el método de cálculo. El score se interpretará
como un indicador experimental relativo al baseline del participante y no como
un diagnóstico.

## 2. Vista general

```text
UnicornLSL
    │ muestras EEG + timestamps
    ▼
Adquisición LSL
    │ canales seleccionados a 250 Hz
    ▼
Buffer circular (4 s)
    │
    ├── control de continuidad y tasa efectiva
    ▼
Preprocesamiento causal
    │ µV → notch 60 Hz → band-pass → referencia → artefactos
    ▼
Ventana espectral (1 s / 250 muestras)
    │ nueva ventana cada 200 ms / 50 muestras
    ▼
PSD mediante Welch
    │ densidad espectral por canal
    ▼
Extractor de características
    │ theta, alfa, beta, potencias relativas y razones
    ▼
Normalizador basal
    │ z-score específico de sujeto y sesión
    ▼
Estimador del índice
    ├── MVP: score espectral explícito
    └── Fase supervisada: Ridge Regression
    ▼
Suavizado temporal
    │ índice estable y con baja latencia
    ▼
Feedback visual/auditivo + registro sincronizado
```

La adquisición nunca debe esperar a que terminen Welch, el modelo o la interfaz.
Un hilo o tarea recibe LSL y escribe en el buffer; otra procesa una copia coherente
de la última ventana; la interfaz únicamente consume el resultado más reciente.

## 3. Adquisición y canales

El módulo de adquisición descubre todos los streams LSL y selecciona de forma
explícita el stream EEG del Unicorn por nombre, tipo, número de canales y
`source_id`. Cada muestra incluye sus valores EEG y un timestamp LSL. También se
registran la frecuencia nominal, la frecuencia realmente recibida y posibles
saltos temporales.

La selección de electrodos debe ser configurable. No conviene heredar
automáticamente `C3, Cz, C4` solo porque eran apropiados para imaginación motora.
Para una primera evaluación de Theta/Beta se propone conservar `Fz` y `Cz` como
canales principales y `C3, C4` como información complementaria, siempre que esa
elección se valide experimentalmente. Si el protocolo exige exclusivamente tres
canales, puede iniciarse con `C3, Cz, C4`, pero debe compararse posteriormente
contra una configuración que incluya `Fz`.

Contrato de salida de adquisición:

```text
samples:    float[n_muestras, n_canales]
timestamps: float[n_muestras]
channels:   lista ordenada de nombres
sfreq:      frecuencia nominal y estimada
units:      µV
```

## 4. Buffer y temporización

Se utilizará un buffer circular de cuatro segundos. A 250 Hz conserva 1000
muestras por canal. El análisis toma siempre el segundo más reciente: 250
muestras. Una nueva evaluación se dispara cada 200 ms, es decir, cada 50 muestras.
Dos ventanas consecutivas se solapan en 800 ms (80 %).

```text
Ventana 1: 0.0 ───────── 1.0 s
Ventana 2:       0.2 ───────── 1.2 s
Ventana 3:             0.4 ───────── 1.4 s
```

El disparo debe basarse preferiblemente en el número de muestras y validarse con
los timestamps LSL. Si faltan datos, se repiten timestamps o la tasa efectiva se
aleja demasiado de la nominal, la ventana se marca como inválida; no se rellena
silenciosamente con valores inventados.

## 5. Preprocesamiento causal

El preprocesamiento reutiliza los principios ya comprobados en
`BCI_Preprocessing`, pero sus parámetros deben almacenarse en una configuración
propia del protocolo:

1. Confirmar que los valores del stream están en µV.
2. Aplicar Notch a 60 Hz para la interferencia de red de Colombia.
3. Aplicar un band-pass causal orientado al análisis espectral. Se propone
   inicialmente 1–40 Hz para conservar theta, alfa y beta completas; utilizar
   8–30 Hz eliminaría parte de theta y haría inválido el cociente Theta/Beta.
4. Aplicar la referencia elegida. CAR solo con pocos electrodos cambia la señal
   espacialmente, por lo que debe compararse con la referencia disponible del
   Unicorn y mantenerse idéntica durante entrenamiento y ejecución.
5. Detectar artefactos por amplitud, inicialmente ±150 µV, además de valores no
   finitos, señal plana y saltos imposibles.

Los filtros son causales, conservan su estado entre chunks y se ejecutan una sola
vez sobre las muestras entrantes. No se debe volver a filtrar desde cero cada
ventana solapada. Una ventana marcada como artefactada no actualiza el modelo ni
el feedback; se conserva temporalmente el último score válido o se muestra una
indicación de mala calidad.

## 6. Estimación espectral mediante Welch

Para cada ventana válida de un segundo se estima la densidad espectral de potencia
(PSD) por canal mediante Welch. Una configuración inicial a 250 Hz es:

```text
window_samples = 250
welch_window   = Hann
nperseg        = 128
noverlap       = 64
nfft           = 256
detrend        = constant
scaling        = density
```

`nperseg=128` permite promediar varios segmentos dentro del segundo y reduce la
variabilidad frente a un único periodograma. `nfft=256` produce bins cercanos a
0.98 Hz, pero el zero-padding no crea resolución física adicional: la resolución
real sigue limitada por la duración de los segmentos. Esta configuración deberá
compararse con `nperseg=250`, que ofrece mejor separación nominal de frecuencias
pero prácticamente no obtiene el beneficio de promediar varios segmentos.

La potencia de cada banda se calcula integrando la PSD, no promediando amplitudes
EEG crudas:

```text
P_banda = integral de PSD(f) dentro de los límites de la banda
```

## 7. Características propuestas

Las bandas iniciales deben fijarse una sola vez para entrenamiento y tiempo real:

```text
Theta: 4–8 Hz
Alpha: 8–13 Hz
Beta:  13–30 Hz
Total: 4–30 Hz
```

Por cada canal se calculan:

- potencia absoluta de theta, alfa y beta;
- potencia relativa de theta, alfa y beta respecto a la potencia total 4–30 Hz;
- `Theta/Beta = P_theta / (P_beta + epsilon)`;
- `Beta/(Theta+Alpha) = P_beta / (P_theta + P_alpha + epsilon)`;
- logaritmo de las potencias y razones cuando reduzca la asimetría;
- agregados entre canales: media, mediana y, opcionalmente, diferencias
  espaciales justificadas.

`epsilon` evita divisiones por cero, pero una potencia extremadamente baja debe
activar además un control de calidad. Todas las fórmulas, límites de banda,
canales y unidades se guardan junto al modelo para impedir diferencias entre el
entrenamiento y la ejecución.

## 8. Baseline y normalización

Al inicio de la sesión se registra un baseline continuo de aproximadamente dos
minutos con ojos abiertos y fijación en una cruz. Los primeros segundos pueden
descartarse para permitir estabilización. El baseline se procesa con exactamente
el mismo pipeline y produce múltiples vectores de características válidos.

Para cada característica se guardan la media y desviación estándar del baseline:

```text
z = (característica_actual - media_baseline) / desviación_baseline
```

Si la desviación es prácticamente cero, la característica se invalida o se usa
un mínimo numérico documentado. La normalización es específica del sujeto y la
sesión. No se recalcula continuamente durante el bloque de neurofeedback porque
eso movería el objetivo mientras el participante intenta modularlo.

Hay dos normalizaciones diferentes que no deben confundirse:

- **Z-score basal:** adapta las características en línea al estado inicial de la
  sesión.
- **Scaler del modelo:** parámetros aprendidos únicamente con los datos de
  entrenamiento para que Ridge reciba variables comparables.

En validación, ambos parámetros deben estimarse sin mirar el conjunto de prueba.

## 9. Dos etapas para construir el índice

### 9.1 MVP sin modelo supervisado

Antes de entrenar Ridge puede probarse el sistema completo con un score explícito.
Como mayor Theta/Beta suele interpretarse en muchos protocolos como menor control
atencional, un índice candidato puede invertir su z-score. Otra alternativa es
usar directamente el z-score de `Beta/(Theta+Alpha)`. Después se limita solamente
la representación visual a 0–100.

Este MVP demuestra latencia, estabilidad, calidad de señal y comprensión del
feedback. No demuestra que el score prediga atención real.

### 9.2 Ridge Regression supervisada

Ridge recibe múltiples características normalizadas y estima una variable
continua:

```text
attention_raw = intercepto + suma(coeficiente_i × característica_i)
```

Ridge es apropiada como primera opción porque controla coeficientes inestables
cuando las potencias y razones están correlacionadas. Sin embargo, necesita una
variable objetivo `y` independiente. No es correcto entrenarla usando
Theta/Beta como objetivo y después afirmar que predice atención; solo aprendería
a reconstruir una fórmula conocida.

Posibles objetivos válidos, definidos antes de registrar los datos, incluyen:

- desempeño continuo o por bloque en una tarea atencional;
- tiempo de reacción y exactitud transformados en un score continuo;
- niveles experimentales de demanda atencional cuidadosamente diseñados;
- una medida externa validada y temporalmente alineada.

La selección de características puede hacerse dentro de la validación mediante
regularización, eliminación de variables casi constantes y comparación de
subconjuntos predefinidos. Nunca se seleccionan características utilizando el
conjunto de prueba final.

## 10. Entrenamiento y validación

Cada registro de entrenamiento debe conservar: sujeto, sesión, bloque, intervalo
temporal, características, calidad de señal y objetivo. Debido al fuerte
solapamiento de las ventanas, no se deben repartir ventanas aleatoriamente entre
train y test: ventanas vecinas compartirían el 80 % de sus muestras y producirían
fuga de información.

La división se realiza por bloques o sesiones completas. Dependiendo del objetivo:

- **Modelo personalizado:** entrenar con sesiones anteriores y probar en una
  sesión posterior del mismo sujeto.
- **Modelo general:** validación dejando sujetos completos fuera.

Dentro del conjunto de entrenamiento se comparan canales, parámetros Welch,
características y el parámetro `alpha` de Ridge. La configuración se elige con
validación cruzada agrupada y se evalúa una sola vez sobre el test reservado.

Métricas:

- **MAE:** error absoluto promedio en las unidades del objetivo.
- **RMSE:** penaliza con mayor fuerza los errores grandes.
- **R²:** proporción de variabilidad explicada; puede ser negativa si el modelo
  es peor que predecir la media.

También se deben informar la correlación, latencia, porcentaje de ventanas
rechazadas y estabilidad entre sesiones, aunque no sustituyen las tres métricas
solicitadas.

## 11. Suavizado temporal y feedback

El score crudo se suaviza después del estimador, no antes de calcular las métricas
offline sin declararlo. Se propone comenzar con una media móvil exponencial:

```text
score_suave[t] = α × score_crudo[t] + (1-α) × score_suave[t-1]
```

Un `α` inicial entre 0.2 y 0.4 puede reducir oscilaciones, pero se seleccionará
midiendo el compromiso entre estabilidad y retardo. El sistema actualiza el
score interno cada 200 ms. La pantalla puede animarse con mayor frecuencia, pero
no debe fingir que existen nuevas estimaciones EEG entre actualizaciones.

La transformación a 0–100 es exclusivamente de presentación. Debe estar
calibrada con el baseline y saturarse en límites razonables para impedir que un
artefacto desplace la esfera o la barra. Si la calidad es mala, el feedback se
congela o se oculta y la interfaz explica el motivo.

## 12. Protocolo experimental

Una sesión propuesta contiene:

1. instrucciones durante 2–3 minutos;
2. baseline inicial de aproximadamente 2 minutos para normalización;
3. entre 10 y 15 ensayos de neurofeedback;
4. cada ensayo: baseline/fijación breve de 5 s, neurofeedback de 30–60 s y
   descanso de 10 s;
5. descanso final de 1–2 minutos.

El baseline inicial de dos minutos calcula los parámetros estables de
normalización. La fijación de cinco segundos al comienzo de cada ensayo marca la
transición y permite observar deriva, pero no reemplaza automáticamente el
baseline de sesión.

Todos los cambios de fase se publican como markers LSL y se registran junto con
EEG crudo, timestamps, características, score crudo, score suavizado, calidad de
señal y eventos de feedback. Esto permite reconstruir y auditar cada decisión.

## 13. Módulos previstos

```text
BCI_Neurofeedback/
├── acquisition/          # descubrimiento LSL, inlet y continuidad temporal
├── preprocessing/        # filtros causales, referencia y artefactos
├── features/             # Welch, bandas, potencias relativas e índices
├── baseline/             # estadísticos y z-score por sesión
├── models/               # scaler, Ridge y metadatos versionados
├── smoothing/            # EMA u otro suavizado causal
├── feedback/             # score, esfera/barra/sonido y calidad de señal
├── protocol/             # baseline, bloques, descansos y markers
├── recording/            # EEG, features, predicciones y configuración
├── training/             # splits agrupados, ajuste y evaluación offline
├── tests/                # señales sintéticas, replay y pruebas de latencia
├── config/               # bandas, canales, filtros y temporización
└── docs/ARCHITECTURE.md
```

El artefacto del modelo debe contener como una sola unidad: orden de canales,
frecuencia de muestreo, filtros esperados, bandas, parámetros Welch, nombres y
orden de características, scaler, Ridge, versión y métricas. El programa debe
rechazar un modelo incompatible en vez de intentar usarlo silenciosamente.

## 14. Secuencia de implementación recomendada

1. Reproducir el pipeline LSL y el preprocesamiento causal con archivos o señales
   guardadas.
2. Implementar Welch y verificarlo con senoidales sintéticas conocidas.
3. Implementar potencias, razones y pruebas numéricas.
4. Registrar el baseline y validar el Z-score.
5. Ejecutar un MVP con índice espectral explícito y EMA.
6. Diseñar la tarea que proporcionará el objetivo continuo independiente.
7. Recolectar datos, entrenar Ridge y validar por sesión/sujeto.
8. congelar el pipeline completo y guardarlo en un artefacto versionado.
9. integrar adquisición, inferencia, control de calidad y feedback.
10. medir latencia de extremo a extremo y realizar pruebas piloto.

## 15. Criterios mínimos de aceptación

- recibe 250 muestras/s de forma estable y conserva timestamps;
- produce una ventana de 250 muestras cada 200 ms;
- identifica correctamente señales sintéticas en theta, alfa y beta;
- no genera score a partir de ventanas artefactadas o incompletas;
- aplica baseline y modelo sin fuga de información;
- completa cada actualización antes del siguiente intervalo de 200 ms;
- registra suficiente información para reproducir el score offline;
- diferencia explícitamente índice experimental, predicción del modelo y score
  mostrado al participante.

## Referencias técnicas

- SciPy documenta `welch` como estimador de PSD basado en periodogramas de
  segmentos solapados y promediados:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
- Documentación de Ridge Regression:
  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- Métricas de regresión de scikit-learn:
  https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics

