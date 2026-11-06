# Bloque 66 — Auditoría del modelo supervisado en desarrollo

Este bloque toma automáticamente el checkpoint elegido en la selección **final**
del 65b y ejecuta inferencia de volumen completo sobre los **289 casos de
entrenamiento/desarrollo**. Produce métricas por clase, confusiones de tejidos y
una galería determinista de errores. Sus resultados son un diagnóstico del
comportamiento en entrenamiento, **no una estimación independiente de rendimiento**.

El 65b completado que precede a este bloque seleccionó la época 45. El código
obtiene la época y el archivo de la evidencia final; no impone ese número ni
sustituye la selección por el último checkpoint.

## Aplicar y comprobar

Desde la raíz del proyecto, con el entorno principal activado y el parche allí:

```powershell
git apply --check .\vista3d-block66-development-audit.patch
git apply .\vista3d-block66-development-audit.patch
python -m pytest -q tests/test_supervised66.py
python .\scripts\run_kits23_supervised66.py --stage prepare
python .\scripts\run_kits23_supervised66.py --stage run --max-cases 3
```

`prepare` comprueba la finalización del 65b, su linaje, la selección, los hashes
existentes y los recibos de caché de desarrollo. Congela un protocolo nuevo en
`configs/experiments/kits23_supervised_block66.json`. No descarga ni ejecuta
inferencia. Mantén los archivos y checkpoints de los bloques anteriores.

`run` llama mediante conda al entorno **vista3d-nv**, el mismo del entrenamiento.
Reutiliza el preprocesado del 65a, los pesos seleccionados y los parámetros de
inferencia del 65b: no cambia resolución, arquitectura, clases ni posprocesado.
La primera llamada hace tres casos nuevos y exporta un ZIP. Comparte su salida y
el ZIP que imprime para verificar el arranque antes de continuar.

Tras comprobar esa primera tanda, completa las fases:

```powershell
python .\scripts\run_kits23_supervised66.py --stage run
python .\scripts\run_kits23_supervised66.py --stage review --max-cases 3
python .\scripts\run_kits23_supervised66.py --stage review
```

La revisión visual usa CPU y las dependencias de visualización del entorno
principal; reutiliza las predicciones guardadas. La inferencia carga el modelo
una vez por invocación con casos pendientes. No inicia otro entrenamiento.
`--max-cases` cuenta **casos nuevos por llamada**, no un objetivo acumulado.
Sin ese argumento se procesan todos los pendientes. Los casos completos se
verifican y reutilizan; repetir una llamada con todo completo no ejecuta inferencia.

Para reconstruir el informe a partir de los artefactos existentes:

```powershell
python .\scripts\run_kits23_supervised66.py --stage report
```

`report` verifica hashes de predicciones y métricas existentes; no vuelve a leer
los volúmenes originales ni recalcula las predicciones. `run` verifica los inputs
originales de los casos reutilizados y los de cada nuevo caso antes/después de
procesarlo. `review` verifica los inputs originales antes/después de renderizar.

## Resultados y reanudación

Directorio nuevo:

```text
outputs/reporting/full489_block66/<audit-id>/
```

- `protocol.json`: checkpoint, cohorte, política y hashes de código/evidencia.
- `cases/<case>/attempt_<id>/prediction.nii.gz`: etiquetas 0–3 en la rejilla original.
- `cases/<case>/attempt_<id>/metrics.json`: métricas, matriz 4×4 y volúmenes físicos.
- `cases/<case>/complete.json`: recibo de finalización y hashes de ambos archivos.
- `progress.json`, `summary.json`: cobertura y agregados de desarrollo.
- `review_selection.json`: selección determinista tras completar toda la cohorte.
- `review/<case>/attempt_<id>/review.png` y su `complete.json`: paneles y trazabilidad.
- `review_progress.json`: cobertura de la galería, independiente de la inferencia.
- `last_action.json`, `export_integrity.json`: estado de ejecución y problemas detectados.
- `report.html`: informe en inglés con tablas y galería desplegable.
- `block66_development_audit_report.zip`: exportación con manifiesto SHA-256.

El ZIP incluye JSON, HTML e imágenes seleccionadas; no incluye pesos, caché ni
NIfTI completos. Extrae el ZIP antes de abrir `report.html`. Los PNG contienen
cortes de CT de los casos revisados.

La finalización de cada caso se publica solo después de guardar los artefactos y
verificar los inputs. Si falla una operación, se detiene esa tanda. Repite la misma
fase: los intentos incompletos no se aceptan como resultados, y se crea un nuevo
intento para el caso pendiente. La alteración de un resultado completado provoca
un error explícito; el código no lo sobreescribe ni omite el caso silenciosamente.
El proceso mantiene el bloqueo de ejecución ya usado por los bloques previos.

Los contadores de una llamada fallida reflejan los casos verificados hasta el
fallo; el exportador vuelve a comprobar todos los recibos disponibles y presenta
su cobertura en `summary.json`. La evidencia de error queda visible en el ZIP.
Los hashes del protocolo del 66 impiden mezclar código, pesos o selección distintos.

## Qué se mide

Las etiquetas son fondo=0, riñón=1, tumor=2 y quiste=3. La matriz tiene **referencia
en filas y predicción en columnas**. Se acumula por bloques de ocho cortes para
evitar una copia entera del volumen con índices int64. El conjunto renal combina
las etiquetas 1–3; una confusión tumor/quiste no es un error de conjunto renal.

Para cada clase se guardan los recuentos, Dice, precisión, sensibilidad y
volúmenes de referencia, predicción, falsos positivos y falsos negativos en mL.
El volumen del vóxel es `abs(det(affine[:3,:3]))/1000`.

- Dice vale 1 si ambas máscaras están vacías; 0 si solo una lo está, igual que en 65.
- Precisión es `null` si no hay predicción; sensibilidad es `null` si no hay referencia.
- Los agregados distinguen todos los casos y los positivos en la referencia.
- Para clases ausentes se informa cuántos casos tienen predicción y su volumen FP.
- La columna de quiste desglosa qué tejido real se predice como quiste.
- La fila de quiste desglosa qué etiqueta recibe el quiste real.

Los tiempos guardados separan inferencia y tiempo de inferencia más recuentos;
excluyen auditoría de archivos y escritura. No constituyen un benchmark de velocidad.
Los volúmenes completos se procesan de uno en uno. La inferencia conserva el
ensamblado de logits en RAM usado en 65; la caché y las predicciones ocupan disco.

## Selección visual

Se unen, sin duplicados y con desempate por identificador, estas reglas:

1. Tres mayores volúmenes FP de quiste en casos sin quistes de referencia.
2. Tres mayores volúmenes FP de quiste en casos con quistes de referencia.
3. Tres menores Dice de tumor entre casos positivos para tumor.
4. Tres menores Dice de quiste entre casos positivos para quiste.
5. Tres menores Dice de conjunto renal entre casos positivos.
6. Dos mayores Dice de quiste entre casos positivos, como controles.

Son **hasta 17 casos distintos**, normalmente menos por coincidencias. Se guardan
todas sus razones; la primera regla determina el foco de la figura. Cada plano
selecciona independientemente el corte con mayor área del foco. Si no hay foco,
se usa el desacuerdo, luego la referencia y finalmente el centro.

Cada figura contiene planos sagital, coronal y axial con CT, referencia,
predicción y errores. Se conserva la proporción física y se rotulan direcciones.
La reorientación a los ejes voxel más próximos a RAS permuta/invierte ejes,
conservando la oblicuidad nativa; no crea cortes anatómicos ortogonales mediante
interpolación. Referencia: https://nipy.org/nibabel/image_orientation.html
La selección está guiada por los resultados; no representa una revisión de todo
el volumen ni una evaluación clínica.

## Interpretación y siguiente decisión

Este bloque no modifica los 80 casos de validación ni abre test120. Lee los
resúmenes de validación ya obtenidos para verificar la elección final del modelo,
pero no hace inferencia ni consulta imágenes/caché de validación o test.

Los resultados de desarrollo no deben compararse directamente con un agregado de
NV en otra cohorte o con otra definición de etiqueta. La galería y las confusiones
permitirán concretar el siguiente cambio experimental, especialmente para quistes,
conservando la separación entre desarrollo, selección en validación y test final.

No se requiere cambiar los scripts de los bloques 64–65b. El parche solo añade
archivos; conserva sus hashes y los protocolos existentes.
