# Bloque 67 — Sensibilidad del filtrado por componentes

Este bloque compara doce filtros fijos con las predicciones originales del 66 en
los **289 casos de desarrollo**. Mide falsos positivos eliminados y daño sobre
tejido correctamente segmentado. Es un experimento exploratorio sobre casos de
entrenamiento: no demuestra rendimiento independiente ni selecciona una variante
para despliegue.

## Aplicación y primera tanda

Con el entorno principal `tfg-vista3d-medical-imaging` activo, desde la raíz:

```powershell
git apply --check .\vista3d-block67-component-sensitivity.patch
git apply .\vista3d-block67-component-sensitivity.patch
python -m pytest -q tests/test_supervised67.py
python .\scripts\run_kits23_supervised67.py --stage prepare
python .\scripts\run_kits23_supervised67.py --stage run --max-cases 3
```

Comparte la salida y el ZIP de esos tres casos para comprobar paridad y tiempo.
`prepare` exige un bloque 66 completo: 289 inferencias y todas las revisiones
seleccionadas, sin errores de integridad. Obtiene la cohorte y el checkpoint de
los protocolos existentes; no impone manualmente la época 45.

Congela `configs/experiments/kits23_supervised_block67.json`: fuente, variantes,
política, código y versiones del entorno CPU. No aplica filtros en `prepare`.
El entorno principal debe conservar NumPy, SciPy y NiBabel; el experimento no
llama a CUDA ni a conda. Las pruebas de integración con los bloques previos
utilizan también PyTorch/MONAI cuando están disponibles.

Tras comprobar la primera tanda, completa los pendientes:

```powershell
python .\scripts\run_kits23_supervised67.py --stage run
```

Se reutilizan los tres casos completos. `--max-cases` limita casos **nuevos por
llamada**; sin ese argumento se procesan todos los pendientes. No hay fase de
entrenamiento ni una fase `review` en este bloque. El informe compara también los
17 casos revisados en el 66, incluidos sus dos controles positivos de quiste.

Para reconstruir el informe a partir de los resultados guardados:

```powershell
python .\scripts\run_kits23_supervised67.py --stage report
```

## Variantes congeladas

El producto de estas opciones da doce variantes, además del resultado original:

| Eje | Opciones |
|---|---|
| Clases que pueden eliminarse | Solo quiste / tumor y quiste |
| Referencia espacial | Todo el riñón predicho / sus dos componentes mayores |
| Distancia máxima | 5 / 10 / 20 mm |

Riñón=1, tumor=2, quiste=3, fondo=0. Los componentes de tumor y quiste se calculan
por separado con **6-conectividad** (vecinos por cara). Se conserva un componente
completo si al menos uno de sus vóxeles está a una distancia menor o igual al
umbral de algún vóxel del riñón de referencia espacial. De lo contrario pasa a
fondo. La tolerancia numérica es 1e-7 mm.

La distancia es euclídea entre **centros de vóxeles en coordenadas físicas**,
usando la transformación afín completa. Admite espaciado anisótropo, rotación y
cizallamiento. No es distancia entre superficies continuas ni HD95.

Los dos componentes mayores del riñón se ordenan por número de vóxeles; los
empates usan el identificador de componente en el orden de recorrido del array.
**No se borra ningún vóxel de riñón**: esta opción restringe únicamente qué
vóxeles sirven como referencia espacial. Si no se predice riñón, todas las
variantes conservan íntegramente la predicción y registran el fallback.

No se impone tamaño mínimo de lesión. Se conservan completas las extensiones de
un componente conectado que alcanza la zona de apoyo; por ello también pueden
persistir falsos positivos unidos por un puente. Componentes reales alejados o
sin un riñón bien predicho pueden perderse. Estas son hipótesis que el experimento
cuantifica, no garantías anatómicas.

Las decisiones del filtro reciben solo la predicción y su afín. La referencia
manual no determina componentes, distancias, umbrales ni regiones de interés.
Se utiliza posteriormente para evaluar los vóxeles conservados/eliminados.

## Métricas y daño

Antes de aceptar un caso se reproduce exactamente la matriz de confusión y las
métricas del 66. Cada variante mantiene un registro de los vóxeles eliminados,
con referencia en filas y predicción original en columnas. Se trasladan esos
recuentos a la columna de fondo para obtener la matriz filtrada.

Esto evita guardar doce NIfTI por caso. Las pruebas comparan los recuentos con
máscaras filtradas materializadas y con las métricas originales del bloque 66.
La función de reconstrucción existe para verificar esa equivalencia; esta etapa
no genera una nueva colección de predicciones que sustituya al modelo elegido.

Se informa, por clase y por caso:

- Dice antes/después y diferencia filtrado menos original.
- Sensibilidad en casos positivos y su cambio.
- Volumen FP eliminado y volumen TP perdido.
- Casos que mejoran/empeoran y pérdidas de Dice o sensibilidad superiores a 0,01.
- Peores cambios individuales, además del promedio de la cohorte.
- Quistes en casos negativos: número de casos con predicción y volumen FP.
- Tejido de referencia no-fondo eliminado, incluida la confusión entre clases.

Un vóxel de tumor predicho sobre quiste real es FP de tumor pero TP del conjunto
renal. Eliminarlo puede mejorar la precisión tumoral y dañar el conjunto renal;
por eso el informe presenta ambas lecturas. Un filtro que solo elimina no puede
recuperar TP ausentes ni corregir una etiqueta tumor/quiste intercambiada.

Las convenciones de vacío son las del 66: Dice=1 si ambas máscaras están vacías,
Dice=0 si solo una lo está; precisión/sensibilidad indefinidas se representan
con `null`. Se distinguen Dice de todos los casos y Dice de los positivos.
Eliminar FP en casos negativos puede mejorar mucho el promedio global de quiste
sin mejorar la segmentación de quistes presentes.

Los volúmenes FP/TP agregados son sumas; Dice y sensibilidad se promedian por
caso. El umbral de pérdida 0,01 es descriptivo, no una regla automática de
aceptación. **Ninguna variante se proclama ganadora**. No se amplía la rejilla de
umbrales automáticamente en función de los resultados.

## Recursos y trazabilidad

Los vecinos se consultan con `scipy.spatial.cKDTree.query`, distancia euclídea,
`eps=0`, un trabajador y consultas de hasta 131072 coordenadas. Se comparten las
consultas entre umbrales y se procesa una clase de componentes cada vez. No se
crea una imagen de distancias float64 de todo el CT ni doce volúmenes filtrados.

Aun así, los mapas de componentes son arrays enteros de volumen completo; los
volúmenes grandes requieren RAM. Se procesa un caso cada vez. Los tiempos
registrados incluyen recuentos y análisis de componentes, pero excluyen lectura,
auditoría de archivos y escritura; no constituyen un benchmark controlado.

Referencias de implementación:

- https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html

Directorio de salida nuevo:

```text
outputs/reporting/full489_block67/<experiment-id>/
```

Contiene `protocol.json`, `progress.json`, `summary.json`, `last_action.json`,
`report.html`, recibos por caso y
`block67_component_sensitivity_report.zip`. Cada caso completo aporta
`cases/<case>/attempt_<id>/sensitivity.json` con sus doce resultados y un recibo
`complete.json`. El ZIP contiene solo JSON/HTML, con manifiesto SHA-256; extrae
el ZIP y abre `report.html` para consultar las tablas desplegables.

Se verifican predicciones y recibos del 66 al congelar, antes/después de la tanda
y alrededor de cada caso nuevo. Los inputs originales de desarrollo se verifican
antes/después del cálculo y al reutilizar casos completos. Los resultados y los
protocolos anteriores permanecen intactos. Los resultados parciales no son una
estimación de toda la cohorte.

Ante una interrupción, repite `run`: los casos incompletos vuelven a intentarse;
los completos se verifican y reutilizan. Una alteración de un artefacto completo
produce error, no sobrescritura. Una llamada fallida puede haber verificado solo
parte de los resultados; el ZIP vuelve a inventariar los recibos existentes y
mantiene el error visible. Conserva la causa del fallo para corregirla antes de
reanudar. Los cambios de código, política o entorno tras el freeze se rechazan.

El código heredado comprueba metadatos y resúmenes históricos de selección del
65b. **No se leen volúmenes ni cachés de validación/test**, ni se generan métricas
nuevas de esas particiones. Test120 permanece sellado. Tras completar el 67,
la revisión de beneficio y daño decidirá si se congela una variante para un
experimento separado de validación o si conviene intervenir en el entrenamiento.
