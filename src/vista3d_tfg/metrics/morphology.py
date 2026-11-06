from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True, slots=True)
class MaskMorphology:
    voxel_count: int
    volume_mm3: float
    volume_ml: float
    component_count: int
    bounding_box_size_mm: tuple[float, float, float]
    bounding_box_diagonal_mm: float


def validate_binary_mask(mask: np.ndarray) -> None:
    if mask.ndim != 3:
        raise ValueError(
            f"Expected a 3D mask, got shape {mask.shape}."
        )

    unique_values = np.unique(mask)

    if not np.all(np.isin(unique_values, [0, 1, False, True])):
        raise ValueError(
            "Mask must be binary."
        )


def validate_spacing(
    spacing_mm: tuple[float, float, float],
) -> None:
    if len(spacing_mm) != 3:
        raise ValueError(
            "Voxel spacing must contain exactly three values."
        )

    if any(value <= 0 for value in spacing_mm):
        raise ValueError(
            "Voxel spacing values must be positive."
        )


def voxel_volume_mm3(
    spacing_mm: tuple[float, float, float],
) -> float:
    validate_spacing(spacing_mm)

    return float(np.prod(spacing_mm))


def mask_volume_mm3(
    mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> float:
    validate_binary_mask(mask)

    return (
        int(np.count_nonzero(mask))
        * voxel_volume_mm3(spacing_mm)
    )


def connected_component_count(mask: np.ndarray) -> int:
    validate_binary_mask(mask)

    _, count = ndimage.label(
        mask.astype(bool),
        structure=ndimage.generate_binary_structure(
            rank=3,
            connectivity=1,
        ),
    )

    return int(count)


def bounding_box_size_mm(
    mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> tuple[float, float, float]:
    validate_binary_mask(mask)
    validate_spacing(spacing_mm)

    coordinates = np.argwhere(mask)

    if coordinates.size == 0:
        return (0.0, 0.0, 0.0)

    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)

    size_voxels = maximum - minimum + 1

    size_mm = size_voxels * np.asarray(
        spacing_mm,
        dtype=float,
    )

    return tuple(
        float(value)
        for value in size_mm
    )


def bounding_box_diagonal_mm(
    mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> float:
    size_mm = bounding_box_size_mm(
        mask,
        spacing_mm,
    )

    return float(
        np.linalg.norm(
            np.asarray(size_mm)
        )
    )


def summarize_binary_mask(
    mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> MaskMorphology:
    validate_binary_mask(mask)
    validate_spacing(spacing_mm)

    voxel_count = int(
        np.count_nonzero(mask)
    )

    volume_mm3 = mask_volume_mm3(
        mask,
        spacing_mm,
    )

    return MaskMorphology(
        voxel_count=voxel_count,
        volume_mm3=volume_mm3,
        volume_ml=volume_mm3 / 1000.0,
        component_count=connected_component_count(mask),
        bounding_box_size_mm=bounding_box_size_mm(
            mask,
            spacing_mm,
        ),
        bounding_box_diagonal_mm=bounding_box_diagonal_mm(
            mask,
            spacing_mm,
        ),
    )

# Importa la anotación de futuros para habilitar la evaluación diferida de tipos.

# Importa el decorador para definir clases de datos inmutables y eficientes.

# Importa la librería NumPy para operaciones numéricas y matriciales eficientes.
# Importa el submódulo ndimage de SciPy para procesamiento de imágenes multidimensionales.

# Define una clase de datos inmutable y optimizada en memoria para almacenar métricas morfológicas de una máscara binaria.
# Almacena el número total de vóxeles activos (diferentes de cero).
# Almacena el volumen total de la máscara expresado en milímetros cúbicos (mm³).
# Almacena el volumen total convertido a mililitros (ml).
# Almacena el número total de componentes conexos detectados en la máscara.
# Almacena las dimensiones físicas de la caja delimitadora (bounding box) en milímetros para cada eje.
# Almacena la longitud de la diagonal principal de la caja delimitadora en milímetros.

# Define una función para validar que la máscara de entrada sea un arreglo tridimensional válido.
# Comprueba si el número de dimensiones de la máscara es estrictamente distinto de 3.
# Lanza un error de valor si la máscara no tiene exactamente tres dimensiones.

# Obtiene los valores únicos presentes en la matriz de la máscara.

# Comprueba si todos los valores únicos de la máscara corresponden exclusivamente a estados binarios válidos (0, 1, False o True).
# Lanza un error de valor si la máscara contiene valores no binarios.

# Define una función para validar la corrección de las dimensiones del espaciado de los vóxeles.
# Verifica que la tupla de espaciado contenga exactamente tres valores numéricos.
# Lanza un error de valor si el espaciado no tiene una longitud de tres elementos.

# Comprueba si alguno de los valores del espaciado es menor o igual a cero.
# Lanza un error de valor si se detectan valores de espaciado no positivos.

# Define una función para calcular el volumen físico de un vóxel individual en milímetros cúbicos.
# Valida previamente que el espaciado de los vóxeles sea correcto.

# Calcula el producto de las dimensiones del espaciado y lo devuelve como un número flotante.

# Define una función para calcular el volumen total de una máscara binaria en milímetros cúbicos.
# Valida que la máscara sea un arreglo binario tridimensional correcto.

# Multiplica el recuento de vóxeles distintos de cero por el volumen individual de cada vóxel.

# Define una función para contar el número de componentes conexos independientes presentes en la máscara binaria.
# Valida que la máscara de entrada sea binaria y tridimensional.

# Etiqueta los componentes conexos utilizando una estructura de vecindad cúbica de rango 3 y conectividad 1.

# Devuelve el número total de componentes conexos detectados convertido a entero.

# Define una función para calcular las dimensiones físicas de la caja delimitadora de la máscara en milímetros.
# Valida que la máscara sea binaria y que el espaciado de los vóxeles sea correcto.

# Obtiene las coordenadas de todos los vóxeles activos en la máscara.

# Comprueba si el arreglo de coordenadas está vacío (no hay vóxeles activos).
# Devuelve una tupla de ceros si la máscara está completamente vacía.

# Encuentra las coordenadas mínimas y máximas de los vóxeles activos a lo largo de cada eje.

# Calcula el tamaño en vóxeles incluyendo el vóxel inicial y final.

# Multiplica el tamaño en vóxeles por el espaciado físico correspondiente transformado a tipo flotante.

# Convierte y devuelve las dimensiones calculadas como una tupla de valores flotantes.

# Define una función para calcular la longitud de la diagonal de la caja delimitadora en milímetros.
# Obtiene las dimensiones físicas de la caja delimitadora llamando a la función correspondiente.

# Calcula y devuelve la norma euclidiana (longitud de la diagonal) a partir de las dimensiones de la caja.

# Define una función principal para resumir todas las propiedades morfológicas de una máscara binaria.
# Valida que la máscara sea binaria y que el espaciado sea correcto.

# Cuenta el número total de vóxeles activos en la máscara.

# Calcula el volumen total en milímetros cúbicos de la máscara.

# Construye y devuelve una instancia de la clase MaskMorphology con todas las métricas calculadas.
# Asigna el recuento total de vóxeles.
# Asigna el volumen en milímetros cúbicos.
# Calcula y asigna el volumen en mililitros dividiendo los milímetros cúbicos entre 1000.
# Calcula y asigna el recuento de componentes conexos.
# Calcula y asigna el tamaño físico de la caja delimitadora.
# Calcula y asigna la diagonal de la caja delimitadora.