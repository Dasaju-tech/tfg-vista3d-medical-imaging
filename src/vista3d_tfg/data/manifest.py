from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from .kits23 import KiTS23Case


def case_to_record(case: KiTS23Case) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "case_number": case.case_number,
        "imaging_path": str(case.imaging_path),
        "segmentation_path": str(case.segmentation_path),
        "instances_path": (
            str(case.instances_path)
            if case.instances_path is not None
            else None
        ),
        "has_instances": case.has_instances,
    }


def build_manifest(
    cases: tuple[KiTS23Case, ...],
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "dataset": "KiTS23",
        "number_of_cases": len(cases),
        "cases": [
            case_to_record(case)
            for case in cases
        ],
    }


def write_manifest(
    manifest: dict[str, object],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # Importa la anotación de futuros para habilitar la evaluación diferida de tipos.

    # Importa la función asdict de dataclasses (aunque no se usa explícitamente en el código actual).
    # Importa el módulo json para la serialización y manipulación de datos en formato JSON.
    # Importa la clase Path para la gestión de rutas del sistema de archivos.
    # Importa la clase KiTS23Case desde el módulo local kits23.

    # Define una función para convertir un objeto KiTS23Case en un diccionario plano de Python apto para serialización.
    # Construye un diccionario mapeando los atributos principales del caso a tipos estándar (como cadenas de texto).
    # Incluye el identificador textual del caso.
    # Incluye el número entero identificador del caso.
    # Convierte la ruta del volumen de imagen a una cadena de texto.
    # Convierte la ruta de la segmentación a una cadena de texto.
    # Convierte la ruta de instancias a cadena si existe, o asigna None en caso contrario.
    # Incluye el indicador booleano de si el caso contiene instancias.

    # Define una función para construir un diccionario de manifiesto global que agrupa múltiples casos del dataset.
    # Define la versión del esquema del manifiesto.
    # Especifica el nombre del conjunto de datos.
    # Calcula y almacena el número total de casos incluidos en el manifiesto.
    # Genera una lista de diccionarios aplicando la función de conversión a cada caso de la tupla.

    # Define una función para escribir el diccionario de manifiesto serializado en un archivo JSON en el disco.
    # Asegura que el directorio padre de la ruta de salida exista, creándolo recursivamente si es necesario.
    # Escribe el contenido en formato JSON con una indentación de dos espacios, ordenando las claves alfabéticamente y usando codificación UTF-8.
    