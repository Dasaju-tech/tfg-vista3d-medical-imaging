# Bloque 64 — Viabilidad de entrenamiento supervisado multiclase

El bloque 63 completo encuentra que el 88,07 % del volumen de falsos positivos
de quiste cae sobre la etiqueta de tumor de KiTS23, en desarrollo. Es una
proporción agrupada por volumen, no el porcentaje de pacientes ni una conclusión
clínica. Esto motiva probar un modelo que aprenda explícitamente las cuatro
etiquetas de referencia, conservando NV-Segment como comparador automático.

Este bloque congela y ejecuta **una prueba técnica de entrenamiento**. Introduce
una U-Net 3D de MONAI con inicialización aleatoria, independiente de los pesos
VISTA3D/NV-Segment. No constituye un fine-tuning de VISTA3D ni el entrenamiento
completo con 289 casos. El checkpoint de prueba no se utilizará para inicializar
el futuro entrenamiento del experimento principal.

## Configuración congelada

| Elemento | Configuración |
|---|---|
| Etiquetas | 0 fondo; 1 riñón; 2 tumor; 3 quiste |
| Arquitectura | U-Net 3D, canales 16/32/64/128, dos unidades residuales, InstanceNorm, sin dropout |
| Parámetros | 1.188.821 en la implementación comprobada |
| Entradas | CT de un canal; orientación cercana a RAS conservando oblicuidad |
| Remuestreo | 1,5 mm por eje; interpolación lineal de CT y vecino más próximo de etiquetas |
| Intensidades | Ventana HU [-200, 300] transformada a [-1, 1] |
| Lote y parche | Un parche de 64 × 64 × 64; 96 mm por eje |
| Casos | Tres de desarrollo; selección SHA-256 fija entre casos con las cuatro clases de referencia |
| Muestreo | Centros fondo/riñón/tumor/quiste, dos ciclos por caso; uniforme dentro de cada clase |
| Pasos | 8 por caso, 24 actualizaciones en total; sin aumentos de datos en esta prueba |
| Pérdida | Dice sin fondo + entropía cruzada de cuatro clases, pesos iguales |
| Optimizador | AdamW, lr 3e-4, weight decay 1e-5, norma máxima del gradiente 12 |
| Precisión | Autocast FP16 en CUDA; pérdida FP32 y GradScaler con escala inicial 128 |
| Recursos | Un volumen cada vez en CPU y un parche en GPU; sin DataLoader multiproceso |
| Semilla | 6401; operaciones deterministas solicitadas con advertencia si no están disponibles |

Para el informe 63 recibido, la selección es `case_00495`, `case_00084` y
`case_00054`, en ese orden. La selección utiliza presencia de clases de referencia,
sin ordenar por Dice ni por errores de predicción. Es una selección para comprobar
el recorrido de las cuatro clases, no una muestra representativa para estimar
calidad. Si desaparece una clase al remuestrear, la prueba falla y no sustituye
silenciosamente el caso ni cambia la resolución.

La GPU objetivo es la RTX 5070 de escritorio del entorno existente. El consumo
real se mide durante la prueba; los contadores del asignador de PyTorch no
equivalen a residencia física en VRAM ni miden por sí solos RAM compartida.
Se rechazan volúmenes de más de 256 millones de vóxeles de origen o 64 millones
remuestreados antes de las asignaciones correspondientes. No se instala ni se
actualiza ninguna dependencia automáticamente.

## Ejecución en PowerShell desde la raíz del proyecto

Requisito: aplicar este parche después del bloque 63 completo y conservar el
protocolo 57 ya comprometido en Git. El directorio de trabajo de los comandos es
la raíz `tfg-vista3d-medical-imaging`; el entorno activo sigue siendo el principal.

```powershell
git apply --check .\vista3d-block64-supervised-smoke.patch
git apply .\vista3d-block64-supervised-smoke.patch
python -m pytest -q
python .\scripts\run_kits23_supervised64.py --stage prepare
python .\scripts\run_kits23_supervised64.py --stage smoke
```

`prepare` audita los 289 informes y recibos del bloque 63, verifica hashes de los
tres CT y referencias seleccionados frente a sus identidades originales y escribe
`configs/experiments/kits23_supervised_block64.json`. No usa GPU. Los identificadores
predeterminados corresponden a los informes de esta conversación; para otra
ejecución se pueden indicar `--run-id` y `--confusion-id` al preparar.

`smoke` abre el protocolo congelado y lanza automáticamente un worker CUDA en
`vista3d-nv`. El worker inserta el `src` de este checkout en su ruta y no exige una
instalación editable en ese entorno. Sus importaciones no requieren las librerías
de métricas/informes del entorno principal. Utiliza Torch, MONAI, NumPy, NiBabel y
SciPy ya presentes; registra sus versiones y la GPU antes de entrenar. No admite
una ejecución CPU que se presente como una prueba GPU superada.

## Qué debe producir

Los resultados se guardan en
`outputs/reporting/full489_block64/<prefijo-del-experimento>/`.

- `protocol.json`: copia del protocolo congelado, particiones, hashes de código,
  entradas seleccionadas y procedencia del bloque 63.
- `attempt_*/runtime.json`: identidad del entorno y comprobación de kernels CUDA.
- `attempt_*/trace.json`: pérdidas, normas de gradiente, clase y centro de cada
  parche; geometría y número de vóxeles por clase después del remuestreo.
- `attempt_*/checkpoint.pt`: pesos locales de la prueba técnica, configuración y
  vínculo con el protocolo. No contiene estado suficiente para reanudar un
  experimento de entrenamiento completo.
- `attempt_*/worker.json`, `complete.json`: resultado y recibo de integridad.
- `block64_supervised_smoke_report.zip`: protocolo, trazas e informes para compartir;
  no incluye CT, máscaras ni pesos del modelo.

La aceptación exige 24 actualizaciones con pérdidas y gradientes finitos, presencia
de las cuatro clases en los objetivos, cambio de pesos, recarga exacta del estado
guardado y conservación de hashes de código/entradas. El `Status: complete`
certifica viabilidad técnica. Las pérdidas son de entrenamiento: no se exige que
disminuyan monótonamente entre parches distintos y no son una métrica de validación.

Si hay un fallo u OOM, se conserva la evidencia y se genera el ZIP de fallo. Repetir
el comando inicia la pequeña prueba desde cero en un intento nuevo, sin sobrescribir
el anterior. Una ejecución ya completada se reutiliza después de comprobar entradas
y todos sus artefactos, incluido el checkpoint. Una interrupción abrupta puede dejar
el candado común de ejecución; se aplica el procedimiento existente del bloque 58
para comprobar que no sigue vivo el proceso antes de retirar un candado abandonado.

## Reserva y siguiente decisión experimental

El protocolo conserva desarrollo289 / validación80 / test120 del bloque 57. En
este bloque solo se leen las imágenes y etiquetas de los tres casos seleccionados
de desarrollo. Se leen identificadores de las particiones reservadas para comprobar
su exclusión, pero no sus imágenes, etiquetas, predicciones, métricas ni QC.
Las anotaciones intervienen en la selección de parches de entrenamiento. No se ha
implementado una inferencia que use recortes basados en la referencia.

Tras revisar el ZIP se fijará otro contrato para entrenar en desarrollo289 y
seleccionar en validación80: presupuesto/épocas, aumentos, inferencia de volumen
completo sin referencia, restauración a la geometría original, criterio de selección
y tratamiento de clases ausentes. Después se congelará el método antes de abrir
test120. El bloque 64 no habilita esas fases ni afirma una mejora frente a NV-Segment.
La separación por paciente/duplicados y el solapamiento de preentrenamiento del
comparador continúan como limitaciones heredadas; no se presenta como evaluación
externa ni puntuación oficial de KiTS23.

## Validación del parche

Las pruebas comprueban la selección por presencia de referencia, reserva de casos,
rechazo de datos/código alterados, conservación de geometría y etiquetas, muestreo
con padding, gestión de fallos y caché, y ZIP sin volúmenes ni pesos. La prueba de
tensores ejecuta 24 pasos reales en CPU con parches reducidos de 16³ y recarga
el checkpoint; se omite si Torch/MONAI no están instalados en el entorno de tests.
También se comprobó forward/backward con el tamaño de producción 64³ en CPU.
La ejecución CUDA real queda por verificar en el ordenador del proyecto.

Validación local de esta entrega: **22 pruebas nuevas superadas**, también con
MONAI 1.5.2. Suite completa: **597 superadas y dos fallos previos** al interpretar
rutas Windows desde Linux en `test_nv_segment_integration.py`. El runtime local
de pruebas de tensores utiliza Torch 2.11.0 CPU; no reproduce CUDA/Torch 2.8.0 de
la estación de trabajo. El worker registrará el runtime real sin actualizarlo.

Referencias de las operaciones utilizadas:
[AMP de PyTorch 2.8](https://docs.pytorch.org/docs/2.8/amp.html) y
[remuestreo de NiBabel](https://nipy.org/nibabel/reference/nibabel.processing.html#nibabel.processing.resample_from_to).

Una vez superada la prueba, versionar los ocho archivos nuevos de este parche y el
JSON generado en `configs/experiments/`; conservar los resultados en `outputs/`
según la política habitual del repositorio. No incluir CT ni checkpoints en Git.
