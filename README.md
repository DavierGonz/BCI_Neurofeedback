# BCI Neurofeedback

Arquitectura propuesta para estimar y presentar en tiempo real un índice continuo
de atención a partir del EEG publicado por el casco Unicorn mediante LSL.

Este proyecto cambia el objetivo del flujo anterior de imaginación motora:

- **Imaginación motora:** clasificar categorías discretas, por ejemplo `LEFT` y
  `RIGHT`, mediante patrones espaciales como CSP.
- **Neurofeedback:** estimar continuamente un estado o índice espectral y
  transformarlo en retroalimentación visual o auditiva.

Por ahora esta carpeta contiene solamente el diseño técnico. No incluye todavía
adquisición, interfaz, entrenamiento ni decisiones clínicas.

## Documento principal

- [`ARCHITECTURE.md`](docs/ARCHITECTURE.md): flujo completo de adquisición,
  procesamiento espectral, normalización, regresión, validación y ejecución en
  tiempo real.
- [`ANALISIS_OFFLINE_MI.md`](docs/ANALISIS_OFFLINE_MI.md): validación offline de
  Theta/Beta y ERD durante imaginación motora para un sujeto.
- `analysis/offline_mi_theta_beta_erd.py`: análisis reproducible desde los GDF
  continuos originales.

## Resultado esperado

Cada 200 ms el sistema utilizará el último segundo de EEG válido para generar:

```text
EEG → PSD → bandas e índices → normalización basal → estimación → suavizado
    → índice continuo de atención → feedback
```

El índice representa una medida experimental relativa al baseline del sujeto.
No debe interpretarse por sí solo como diagnóstico ni como medición clínica de
la atención.
