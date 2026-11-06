# Bloque 65b: recuperación trazable del entrenamiento AMP

La ejecución se detuvo durante la época 7 al intentar recortar una norma de
gradientes no finita. El código anterior lanzaba la excepción antes de
`GradScaler.step()` y `GradScaler.update()`. Eso impide que el escalador responda
a un desbordamiento de FP16. El log, sin los tensores ni el estado del escalador
del intento fallido, no permite asegurar que el desbordamiento sea la única causa.

Esta corrección añade una continuación **con una identidad nueva**, congelada en
`configs/experiments/kits23_supervised_block65b.json`. Conserva los archivos del
bloque 65/65a, su protocolo y sus resultados. Lee el último checkpoint completo
del padre (se espera la época 6 en esta ejecución), incluidos modelo, AdamW,
GradScaler y estados RNG. Repite la época 7 entera; no recupera sus pasos parciales.

La caché existente se reutiliza desde su ubicación original, verificando entradas,
hashes y versiones de ejecución. No hay descargas, remuestreo ni copias de la caché.
El nuevo directorio está en `outputs/reporting/full489_block65/<nuevo_hash>/`.

## Corrección numérica

- Se comprueba que los gradientes desescalados sean finitos antes del recorte.
- Si hay Inf/NaN y AMP está activo, `GradScaler` omite la actualización y reduce
  la escala. Se vuelve a evaluar **el mismo parche aumentado**, sin cambiar caso,
  etiquetas ni muestreo. El modelo congelado no tiene dropout ni estadísticas
  de normalización acumuladas.
- Se permiten como máximo ocho reintentos tras el intento inicial. Una pérdida
  no finita, un gradiente cero, un fallo persistente o una norma no finita con
  gradientes finitos siguen deteniendo el proceso.
- Se conserva `error_if_nonfinite=True`; no se actualizan pesos con NaN/Inf.
- Los intentos fallidos no cuentan como pasos del optimizador. Una época completa
  sigue teniendo 2.312 actualizaciones efectivas; la época 10 acumula 23.120.
- Cada backoff se guarda en `amp_events/`; los checkpoints nuevos incluyen
  `amp_retries`, escalas inicial/final y número de actualizaciones efectivas.
  Los eventos de un intento interrumpido se conservan también y pueden repetirse
  al repetir una época; el contador del checkpoint corresponde al intento exitoso.

Referencia del comportamiento del escalador y del recorte:
[PyTorch 2.8: Automatic Mixed Precision examples](https://docs.pytorch.org/docs/2.8/notes/amp_examples.html).

No se cambia arquitectura, resolución, parches, pérdida, optimizador, calendario,
particiones ni criterio de selección. La continuación hereda las validaciones ya
completas del padre y registra el hash del experimento y del checkpoint seleccionado,
que puede seguir perteneciendo al padre. Los 120 casos de test permanecen reservados.
No se presenta esta recuperación como una ejecución ininterrumpida del código antiguo.

## Aplicación y continuación en PowerShell

Guardar el parche en la raíz del repositorio:

```powershell
git apply --check .\vista3d-block65b-amp-recovery.patch
git apply .\vista3d-block65b-amp-recovery.patch
python -m pytest -q tests/test_supervised65b.py
python .\scripts\run_kits23_supervised65b.py --stage prepare
python .\scripts\run_kits23_supervised65b.py --stage train --until-epoch 10
```

Continuar con `train` cuando `prepare` muestre `continuation freeze PASSED` y
`Resume from parent epoch: 6; next: 7`. El número se obtiene de los checkpoints
verificados, no del último texto de consola. No borrar protocolos, cachés ni épocas.
Una vez congelada esta continuación, usar el ejecutor **65b**; hacer avanzar el padre
65 invalidaría el historial fijado para esta continuación.

`--until-epoch 10` es un objetivo absoluto: puede repetirse tras una interrupción.
Una época incompleta se repite desde el último checkpoint; si quedó pendiente su
validación, se completa antes de avanzar. Si ya se alcanzó la época 10, no añade otra.
Para continuar más adelante se puede usar el mismo ejecutor con otro objetivo
hasta 50, respetando el calendario de validación ya congelado.

Compartir el ZIP nuevo que indique la consola, llamado
`block65b_supervised_training_report.zip`. Incluye el protocolo nuevo, los eventos
AMP, los resultados nuevos y los recibos/resúmenes congelados del padre. Los pesos,
predicciones y matrices permanecen locales y sus hashes permiten comprobarlos.

## Verificación local

Las pruebas ejercitan el `GradScaler` real en CPU con gradientes Inf/NaN inducidos:
backoff sin actualizar pesos, reintento del mismo parche, límite de reintentos,
paridad exacta de la ruta finita, restauración de modelo/optimizador/RNG, reanudación
desde padre e hijo, validación pendiente, objetivo absoluto, conservación de los
archivos del padre y rechazo de cambios de código, parámetros o checkpoints.
No equivalen a reproducir el fallo con los volúmenes del usuario en una RTX 5070.
