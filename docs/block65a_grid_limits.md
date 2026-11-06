# Bloque 65a — Límites de volumen derivados de la cohorte

Corrige el bloqueo de `prepare` ante `case_00223` sin excluir casos ni modificar
resolución, arquitectura, parches, muestreo, particiones o criterio de selección.
El mensaje anterior no distinguía cuál de los dos límites se había superado;
no disponemos aquí de la cabecera real de ese caso para cuantificarlo.

El bloque 65 usaba topes fijos de 512 millones de vóxeles originales y 64 millones
remuestreados. Ahora `prepare` lee las cabeceras de todos los 289+80 casos, conserva
sus formas en `grid_audit` y fija ambos máximos a los mayores valores observados.
Se mantienen controles de geometría positiva tridimensional y tamaño direccionable,
hashes de entradas, congelación del protocolo y comprobación de límites en la caché.
La estimación de disco se calcula para la cohorte completa. Las formas del informe
de cabeceras usan el orden original de ejes; la caché reorienta a RAS y puede permutarlas,
conservando el mismo producto de dimensiones.

Los máximos son cotas de los datos incluidos, no una certificación de RAM disponible.
Los volúmenes grandes requieren más RAM en preprocesamiento y acumulación de
validación. No se reduce automáticamente la resolución ni se omiten casos ante
falta de memoria. Se imprime el caso que determina cada máximo para poder diagnosticarlo.

En la ejecución comunicada, el fallo sucedió antes de escribir el protocolo 65
y antes de construir caché o entrenar. No hay que borrar JSON ni repetir descargas.
Este parche se aplica sobre el bloque 65 original. No migra ni sobrescribe un
protocolo 65 que ya se hubiera congelado correctamente en otra ejecución.

```powershell
git apply --check .\vista3d-block65a-grid-limits.patch
git apply .\vista3d-block65a-grid-limits.patch
python -m pytest -q tests/test_supervised65.py tests/test_supervised65a_grid_limits.py
python .\scripts\run_kits23_supervised65.py --stage prepare
```

Después de `protocol freeze PASSED`, continuar con las dos cachés y la primera época:

```powershell
python .\scripts\run_kits23_supervised65.py --stage cache --partition development
python .\scripts\run_kits23_supervised65.py --stage cache --partition validation
python .\scripts\run_kits23_supervised65.py --stage train --epochs 1
```

Las pruebas incluyen cabeceras sintéticas por encima de ambos límites antiguos sin
reservar matrices gigantes, un caso posterior mayor, conservación de hiperparámetros,
rechazo de formas inválidas y rechazo de un cambio manual del máximo congelado.
