# Bloque 65 — Entrenamiento supervisado y selección en validación

Este bloque pasa de la prueba técnica del bloque 64 al entrenamiento con los
**289 casos de desarrollo**. Introduce caché de preprocesamiento, checkpoints
reanudables y evaluación de volumen completo en los **80 casos de validación**.
Los **120 casos de test** permanecen reservados. No modifica archivos de bloques
anteriores ni sus predicciones, evaluadores, pesos o recibos.

La U-Net se inicializa desde cero con semilla 6501. El checkpoint del bloque 64
se verifica como evidencia de viabilidad, pero sus pesos no se cargan. La U-Net
es un comparador supervisado independiente; no es fine-tuning de VISTA3D ni una
reproducción de una puntuación oficial KiTS23.

## Protocolo fijo

| Elemento | Decisión |
|---|---|
| Particiones | Desarrollo289 / validación80 / test120, sin redistribución |
| Modelo | U-Net 3D MONAI, canales 16/32/64/128, dos unidades residuales, InstanceNorm |
| Clases | 0 fondo; 1 riñón; 2 tumor; 3 quiste |
| Resolución | 1,5 mm por eje, orientación próxima a RAS, conservando oblicuidad |
| CT | Interpolación lineal; HU [-200,300] normalizados a [-1,1] |
| Referencia de entrenamiento | Vecino más próximo; clases ausentes permitidas |
| Parche | 96³ vóxeles, 144 mm por eje; lote de uno |
| Época | Una permutación de los 289 casos, ocho parches por caso |
| Presupuesto | 50 épocas, 2.312 pasos por época, 115.600 pasos en total |
| Muestreo | Clase uniforme entre las presentes en el caso, vóxel uniforme de esa clase |
| Aumentos | Flip conjunto imagen/etiqueta, p=0,5 por eje; escala [0,9,1,1] y desplazamiento [-0,1,0,1] de intensidad |
| Pérdida | Dice sin fondo + entropía cruzada de cuatro clases, ambas con peso uno |
| Optimizador | AdamW, lr inicial 3e-4, weight decay 1e-5; norma máxima del gradiente 12 |
| LR | `base_lr * 0.5 * (1 + cos(pi * (epoch-1) / 50))` |
| Precisión | Autocast CUDA FP16; pérdida FP32; GradScaler con escala inicial 128 |
| Validación | Después de las épocas 1, 5, 10, 15, 20, 25, 30, 35, 40, 45 y 50 |
| Selección | Mayor media del Dice de riñón/tumor/quiste por caso, promediada sobre los 80 casos; empate: época anterior |
| Parada | Presupuesto fijo; sin early stopping ni cambio automático de hiperparámetros |

La selección es provisional mientras falten épocas. Para cada clase, dos máscaras
vacías reciben Dice 1; si solo una está vacía, Dice 0. Esta convención puede elevar
la media cuando abundan casos sin quistes. Por ello se publican también Dice solo
en referencias positivas, número de referencias negativas con predicción y su
volumen falso positivo, además del Dice de la envolvente renal. No se utiliza Dice
de parches de entrenamiento como criterio de selección.

## Ejecución en PowerShell

Desde la raíz del proyecto, con el entorno principal activo y el bloque 64
completado. Ejecutar cada fase después de que la anterior termine correctamente.

```powershell
git apply --check .\vista3d-block65-supervised-training.patch
git apply .\vista3d-block65-supervised-training.patch
python -m pytest -q
```

Adquirir las imágenes y referencias de los 80 casos de validación con el runner
existente. Esta operación realiza descarga y auditoría de entrada, sin inferencia
de NV-Segment ni cálculo de resultados. Los 289 de desarrollo ya están adquiridos.

```powershell
python .\scripts\run_nv_segment_ct_full489.py --stage acquire --partition validation
python .\scripts\run_kits23_supervised65.py --stage prepare
```

`prepare` verifica el bloque 64, congela hashes de los 369 pares CT/referencia y
el código/método, comprueba duplicados de archivos CT por bytes y estima espacio
de caché leyendo cabeceras. Escribe
`configs/experiments/kits23_supervised_block65.json`. No abre resultados de test.
La adquisición debe estar completa antes de preparar; no se sustituyen casos que
fallen. La comprobación de duplicados por bytes no es una auditoría completa de
identidad de paciente ni detecta todas las duplicaciones de vóxeles.

La caché de desarrollo se estima en **27,8 GiB** usando las geometrías del informe
63. Se suma la caché de los 80 CT de validación. `prepare` imprime la estimación
total y el espacio libre. Las matrices se guardan sin comprimir para acceder a
parches mediante memoria mapeada; su tamaño puede ser mayor que los NIfTI originales.
Es conveniente disponer de espacio adicional para checkpoints y predicciones de
validación. La construcción de cada caso exige además 2 GiB libres de reserva.

Crear las dos cachés; el script selecciona automáticamente `vista3d-nv` para
mantener iguales las versiones de NumPy, NiBabel y SciPy usadas por el worker:

```powershell
python .\scripts\run_kits23_supervised65.py --stage cache --partition development
python .\scripts\run_kits23_supervised65.py --stage cache --partition validation
```

La primera fase guarda CT y etiquetas de entrenamiento, además de recuentos por
clase y corte para muestrear sin recorrer de nuevo toda la máscara. La segunda
guarda CT y geometría. Las referencias de validación se comprueban como entradas,
pero no determinan normalización, recortes ni ventanas de inferencia. No se
almacena una caché de etiquetas de validación.

La caché permite limitar nuevos casos con `--max-cases 10` y continuar repitiendo
el comando sin ese límite. Los casos ya completos se verifican y reutilizan.
`cache_*_progress.json` incluye tamaño, geometría, recibos y clases perdidas por
remuestreo, si las hay. La ausencia de quistes o tumores en un caso es válida; la
desaparición de todo el primer plano provoca error. No se cambia la resolución
automáticamente ni se excluye el caso. Los límites de entrada son 512 millones de
vóxeles originales y 64 millones remuestreados. Incluyen los dos casos de desarrollo
que superaban los 256 millones de la pequeña prueba del bloque 64.

**Primera ejecución: una época completa y validación80.**

```powershell
python .\scripts\run_kits23_supervised65.py --stage train --epochs 1
```

Debe terminar con `Status: partial`, `Completed epochs: 1/50` y 2.312 pasos.
`partial` significa que queda presupuesto del experimento, no que se hayan omitido
casos de esa época. Se requiere completar las dos cachés antes de entrenar.

Compartir el `block65_supervised_training_report.zip` indicado por el comando.
Permite revisar pérdidas, tiempos, memoria, métricas de validación y reanudación.
El ZIP no contiene CT, máscaras, matrices de caché ni checkpoints.

Tras revisar la primera época, el mismo ejecutable permite continuar todo lo
pendiente o avanzar por tandas. `--epochs` es el máximo de épocas **adicionales**
de esa llamada; no cambia el presupuesto total congelado de 50:

```powershell
python .\scripts\run_kits23_supervised65.py --stage train --epochs 49
```

También se puede usar `--epochs 5` repetidamente. Una ejecución que ya haya
alcanzado 50 épocas no entrena más. Para regenerar el ZIP sin GPU:

```powershell
python .\scripts\run_kits23_supervised65.py --stage report
```

## Reanudación e integridad

Los checkpoints completos contienen modelo, AdamW, GradScaler, estados aleatorios
de Torch CPU/CUDA, época y estadísticas. Se guarda una inicialización propia como
época cero. Cada checkpoint tiene nombre único; el recibo se publica después de
guardar, recargar y comprobar los tensores. Nunca se sobrescribe una época completa.

Si se interrumpe una época a mitad, se repite **esa época** desde el checkpoint
anterior. No se promete reanudar en el último parche. Si la interrupción ocurre
durante validación, la época ya está guardada: se reutilizan los casos de validación
con recibo completo y se terminan los pendientes antes de avanzar. Una llamada
con `--epochs 1` puede terminar esa validación pendiente y añadir otra época.

Los datos y la caché se verifican por hash al iniciar cada worker de entrenamiento.
Durante su reutilización en ese proceso se comprueban tamaño y fecha de modificación
de los archivos. Cambios de runtime, configuración, código o contenido detectados
provocan error; no se abre una ejecución alternativa silenciosamente. Las semillas
y estados permiten reproducir el recorrido, pero el aviso `warn_only=True` de CUDA
del bloque 64 sigue impidiendo prometer igualdad bit a bit entre entrenamientos GPU.

`Ctrl+C` usa el mecanismo existente de terminación del árbol de procesos. Un fallo
abrupto del sistema puede dejar el candado común de ejecución; seguir el procedimiento
del bloque 58 para comprobar el proceso antes de retirar un candado abandonado.

## Validación de volumen completo

MONAI recorre todo el CT remuestreado con ventanas 96³, dos ventanas por llamada,
solapamiento 0,25 y combinación gaussiana de logits. Solo las ventanas van a CUDA;
la acumulación se hace en CPU con float32. No hay recorte por la referencia,
test-time augmentation ni posprocesamiento. Tras argmax, la máscara discreta vuelve
por vecino más próximo a la forma y affine del CT original; allí se calculan los
recuentos y Dice. Las máscaras se conservan localmente con sus recibos.

Esta primera implementación utiliza CPU/RAM para la acumulación de volumen completo;
los picos de memoria publicados en estadísticas de época corresponden al entrenamiento,
no al pico completo de RAM ni a toda la fase de validación. No se calculan HD95 ni
Surface Dice durante la selección para no repetir trabajo de superficies en cada
checkpoint. Tampoco se afirma una mejora frente a NV-Segment comparando métricas
de particiones distintas. La evaluación final comparativa requerirá su propio paso
una vez elegido y congelado el método, antes de abrir test120.

## Pruebas de esta entrega

Se comprueban congelación, exclusión de test, rechazo de duplicados por bytes,
casos sin quistes, caché parcial y corrupción, muestreo/aumentos alineados,
convención de clases vacías, inferencia sobre todo el volumen y restauración de
geometría. Dos pruebas ejecutan entrenamiento real con tensores pequeños en CPU:
reanudación equivalente a entrenamiento continuo e interrupción a mitad de época.
La red también pasó forward/backward con parches 96³ y MONAI 1.5.2 en CPU.

La suite completa obtuvo **614 pruebas superadas** y los dos fallos previos de
`test_nv_segment_integration.py` que interpretan rutas Windows desde Linux.
Las **17 pruebas nuevas** pasaron con MONAI 1.5.2 y Torch 2.11.0 CPU. Las pruebas
de tensores se omiten si Torch/MONAI no están en el entorno principal de tests.
CUDA/Torch 2.8.0 se ejecutarán en la estación de trabajo; no se han simulado sus
tiempos ni se ha entrenado aquí con las imágenes reales.

Guardar en Git los archivos nuevos del parche y el JSON generado de configuración.
Mantener datos, caché, checkpoints y predicciones fuera de Git según el procedimiento
habitual del proyecto.
