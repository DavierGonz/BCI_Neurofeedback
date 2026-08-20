# Análisis offline de Theta/Beta y ERD en imaginación motora

## Propósito

Este análisis utiliza un sujeto del dataset empleado anteriormente para
clasificación LEFT/RIGHT. Su objetivo no es entrenar todavía el sistema de
neurofeedback, sino comprobar si las características espectrales cambian entre
un período basal y la imaginación motora.

El sujeto inicial es `A18`, seleccionado anteriormente como el mejor sujeto para
el modelo CSP+PSD+LDA. Tiene seis sesiones, con 40 trials por sesión: 120 LEFT y
120 RIGHT.

## Por qué no se usa `all_epochs_clean-epo.fif`

Ese archivo contiene solamente 0.5–3.5 s después del cue y ya está filtrado entre
8–30 Hz. Por lo tanto, no conserva el baseline y tampoco permite estimar
correctamente theta de 4–8 Hz. El análisis regresa a los `.gdf` continuos
originales sin modificar los archivos existentes.

## Ventanas temporales

Los tiempos del ejemplo recibido, baseline 1–4 s y MI 5–9 s, no coinciden con la
temporización real de este dataset. Se utiliza el cue LEFT/RIGHT como `t=0`:

```text
Baseline: −3 a −1 s
MI:       +1 a +4 s
```

La ventana MI evita el primer segundo después del cue, como fue solicitado, y
permanece dentro del trial real.

## Procesamiento

```text
GDF continuo
→ tipos correctos para EEG/EOG/EMG
→ Notch 50 Hz
→ band-pass 1–40 Hz
→ referencia promedio con los canales EEG
→ selección C3, Cz y C4
→ rechazo de trial si supera ±150 µV
→ PSD Welch
```

Se utiliza Notch de 50 Hz porque el dataset fue registrado en Francia. El
band-pass conserva theta, μ y beta. Welch emplea segmentos Hann de un segundo con
50 % de solapamiento en ambas ventanas, para mantener parámetros comparables.

## Cálculos

```text
Theta = 4–8 Hz
Mu    = 8–13 Hz
Beta  = 13–30 Hz

Theta/Beta = potencia_theta / potencia_beta

Cambio θ/β (%) =
    100 × (θ/β_MI − θ/β_baseline) / θ/β_baseline

ERD (%) =
    100 × (potencia_baseline − potencia_MI) / potencia_baseline
```

Con esta convención, un ERD positivo significa que la potencia disminuyó durante
MI. Un valor negativo representa un aumento de potencia o ERS.

El archivo por canal conserva C3, Cz y C4 individualmente. Para el resumen por
trial se promedian primero las potencias de los tres canales y luego se calculan
las razones; no se promedian cocientes con denominadores diferentes.

## Ejecución

Desde el ambiente de `BCI_Exploration`, que ya contiene MNE y SciPy:

```powershell
cd D:\Proyects\Python\BCI_workspace
.\BCI_Exploration\.venv\Scripts\python.exe `
  .\BCI_Neurofeedback\analysis\offline_mi_theta_beta_erd.py `
  --subject A18
```

Para generar resultados sin abrir la figura:

```powershell
.\BCI_Exploration\.venv\Scripts\python.exe `
  .\BCI_Neurofeedback\analysis\offline_mi_theta_beta_erd.py `
  --subject A18 --no-show
```

## Resultados

En `results/A18/` se generan:

- `theta_beta_erd_por_canal.csv`: una fila por trial y canal;
- `theta_beta_erd_por_trial.csv`: resumen de C3, Cz y C4 por trial;
- `resumen_theta_beta_erd.png`: figura de θ/β, cambio porcentual y ERD;
- `resumen.json`: parámetros, conteos, medias, medianas y prueba emparejada.

La prueba de Wilcoxon indica si existe una diferencia emparejada entre los ratios
de los trials, pero el valor p no mide por sí solo la magnitud ni la relevancia
fisiológica del cambio. Debe interpretarse junto con las distribuciones y ERD.

## Limitación principal: es un dataset de MI, no de neurofeedback

Este dataset fue diseñado para imaginar movimientos LEFT y RIGHT y entrenar un
clasificador de imaginación motora. No fue diseñado para que los participantes
modularan voluntariamente su atención, recibieran retroalimentación continua ni
aprendieran a modificar la razón Theta/Beta. En consecuencia, no existe una razón
experimental para esperar que Theta/Beta cambie de forma global y consistente
como ocurriría en un protocolo específico de neurofeedback atencional.

La imaginación motora se caracteriza principalmente por cambios espaciales en μ
y beta sobre la corteza sensoriomotora. C3 y C4 pueden responder de forma distinta
según la mano imaginada. Al combinar C3, Cz y C4 en un promedio, una reducción
contralateral puede compensarse con una respuesta diferente en otro canal. Un
modelo CSP puede clasificar LEFT y RIGHT aprovechando esas diferencias espaciales
aunque el promedio global de Theta/Beta o ERD sea pequeño.

Por lo tanto, este análisis permite:

- comprobar que la extracción de PSD, Theta/Beta y ERD funciona;
- explorar cambios incidentales entre baseline y MI;
- identificar diferencias por clase, canal, trial y sesión;
- preparar la metodología que después se aplicará al neurofeedback.

Este análisis **no permite** concluir si Theta/Beta es un índice válido de
atención ni si el futuro neurofeedback será eficaz. Esa validación necesitará un
dataset registrado con baseline estable, tareas atencionales, feedback continuo
y una medida externa de desempeño o atención.

La interpretación correcta de un resultado sin cambio global es que el marcador
no fue consistente dentro de este protocolo de MI. No significa que el pipeline
esté defectuoso ni demuestra que Theta/Beta no pueda utilizarse en un experimento
de neurofeedback diseñado específicamente para ese objetivo.
