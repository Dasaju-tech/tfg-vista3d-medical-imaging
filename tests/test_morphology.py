import numpy as np
import pytest

from vista3d_tfg.metrics.morphology import (
    bounding_box_size_mm,
    connected_component_count,
    mask_volume_mm3,
    summarize_binary_mask,
    voxel_volume_mm3,
)


def test_voxel_volume() -> None:
    spacing = (1.0, 2.0, 3.0)

    assert voxel_volume_mm3(spacing) == pytest.approx(6.0)


def test_mask_volume() -> None:
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1

    volume = mask_volume_mm3(
        mask,
        spacing_mm=(1.0, 1.0, 2.0),
    )

    assert np.count_nonzero(mask) == 8
    assert volume == pytest.approx(16.0)


def test_connected_components() -> None:
    mask = np.zeros((10, 10, 10), dtype=np.uint8)

    mask[1:3, 1:3, 1:3] = 1
    mask[7:9, 7:9, 7:9] = 1

    assert connected_component_count(mask) == 2


def test_bounding_box_size() -> None:
    mask = np.zeros((10, 10, 10), dtype=np.uint8)

    mask[2:5, 3:7, 1:6] = 1

    result = bounding_box_size_mm(
        mask,
        spacing_mm=(1.0, 2.0, 3.0),
    )

    assert result == pytest.approx(
        (3.0, 8.0, 15.0)
    )


def test_empty_mask() -> None:
    mask = np.zeros((5, 5, 5), dtype=np.uint8)

    summary = summarize_binary_mask(
        mask,
        spacing_mm=(1.0, 1.0, 1.0),
    )

    assert summary.voxel_count == 0
    assert summary.volume_mm3 == 0.0
    assert summary.volume_ml == 0.0
    assert summary.component_count == 0
    assert summary.bounding_box_size_mm == (0.0, 0.0, 0.0)
    assert summary.bounding_box_diagonal_mm == 0.0


def test_non_binary_mask_is_rejected() -> None:
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    mask[1, 1, 1] = 2

    with pytest.raises(ValueError):
        summarize_binary_mask(
            mask,
            spacing_mm=(1.0, 1.0, 1.0),
        )

# Importa la librería NumPy para la creación y manipulación de arreglos numéricos.
# Importa la librería pytest para definir y ejecutar las pruebas unitarias.
# Importa las funciones de cálculo morfológico y métricas desde el módulo correspondiente del proyecto.

# Define una prueba unitaria para comprobar el cálculo correcto del volumen individual de un vóxel.
# Define una tupla con el espaciado físico de los vóxeles en los tres ejes.

# Afirma que el volumen calculado del vóxel coincide con el producto esperado utilizando una aproximación numérica de pytest.

# Define una prueba unitaria para verificar el cálculo del volumen total de una máscara binaria.
# Crea una matriz tridimensional de ceros de tamaño 5x5x5 con tipo entero sin signo.
# Rinde una región específica de la matriz con valor 1 para simular una estructura activa.

# Calcula el volumen de la máscara utilizando un espaciado de vóxeles definido.

# Comprueba que la cantidad de elementos distintos de cero en la máscara sea la esperada (8 vóxeles).
# Afirma que el volumen total calculado sea el correcto usando aproximación flotante.

# Define una prueba unitaria para comprobar el recuento correcto de componentes conexos separados.
# Crea una matriz tridimensional de ceros de tamaño 10x10x10.

# Define dos regiones activas separadas dentro de la matriz para simular dos componentes distintos.
# Define el primer bloque activo.
# Define el segundo bloque activo separado del anterior.

# Afirma que la función de recuento detecte exactamente dos componentes conexos independientes.

# Define una prueba unitaria para comprobar el cálculo correcto del tamaño de la caja delimitadora en milímetros.
# Crea una matriz tridimensional de ceros de tamaño 10x10x10.

# Rellena una región específica de la matriz con el valor 1.

# Calcula el tamaño físico de la caja delimitadora usando un espaciado de vóxeles específico para cada eje.

# Afirma que las dimensiones físicas resultantes de la caja delimitadora coincidan con los valores esperados.

# Define una prueba unitaria para verificar el comportamiento del resumen morfológico ante una máscara completamente vacía.
# Crea una matriz tridimensional vacía de ceros.

# Genera el resumen morfológico completo de la máscara vacía utilizando un espaciado unitario.

# Afirma que el recuento de vóxeles activos sea cero.
# Afirma que el volumen en milímetros cúbicos sea cero.
# Afirma que el volumen en mililitros sea cero.
# Afirma que el recuento de componentes sea cero.
# Afirma que las dimensiones de la caja delimitadora sean una tupla de ceros.
# Afirma que la diagonal de la caja delimitadora sea cero.

# Define una prueba unitaria para verificar que se rechacen correctamente las máscaras que contengan valores no binarios.
# Crea una matriz tridimensional vacía y asigna un valor inválido (2) en una posición para romper la condición binaria.

# Utiliza un contexto de prueba para asegurar que se lance una excepción de tipo ValueError al intentar resumir la máscara no binaria.