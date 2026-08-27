from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


CASE_PATTERN = re.compile(r"^case_(\d+)$")


@dataclass(frozen=True, slots=True)
class KiTS23Case:
    case_id: str
    case_number: int
    directory: Path
    imaging_path: Path
    segmentation_path: Path
    instances_path: Path | None

    @property
    def has_instances(self) -> bool:
        return self.instances_path is not None


def parse_case_number(directory_name: str) -> int:
    match = CASE_PATTERN.fullmatch(directory_name)

    if match is None:
        raise ValueError(
            f"Invalid KiTS23 case directory name: {directory_name!r}"
        )

    return int(match.group(1))


def discover_case(case_directory: Path) -> KiTS23Case:
    case_directory = case_directory.resolve()

    case_number = parse_case_number(case_directory.name)

    imaging_path = case_directory / "imaging.nii.gz"
    segmentation_path = case_directory / "segmentation.nii.gz"
    instances_directory = case_directory / "instances"

    if not imaging_path.is_file():
        raise FileNotFoundError(
            f"Missing imaging volume: {imaging_path}"
        )

    if not segmentation_path.is_file():
        raise FileNotFoundError(
            f"Missing reference segmentation: {segmentation_path}"
        )

    return KiTS23Case(
        case_id=case_directory.name,
        case_number=case_number,
        directory=case_directory,
        imaging_path=imaging_path,
        segmentation_path=segmentation_path,
        instances_path=(
            instances_directory
            if instances_directory.is_dir()
            else None
        ),
    )


def discover_cases(dataset_root: Path) -> tuple[KiTS23Case, ...]:
    dataset_root = dataset_root.resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"KiTS23 dataset root does not exist: {dataset_root}"
        )

    if not dataset_root.is_dir():
        raise NotADirectoryError(
            f"KiTS23 dataset root is not a directory: {dataset_root}"
        )

    candidates = [
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and CASE_PATTERN.fullmatch(path.name)
    ]

    cases = [discover_case(path) for path in candidates]

    return tuple(
        sorted(
            cases,
            key=lambda case: case.case_number,
        )
    )

# Importa la anotación de futuros para permitir la evaluación diferida de tipos en Python.

# Importa el decorador para crear clases de datos eficientes e inmutables.
# Importa la clase Path para la gestión moderna de rutas de archivos.
# Importa el módulo de expresiones regulares para validar nombres y extraer identificadores.

# Define un patrón de expresión regular compilado para identificar directorios de casos que sigan exactamente el formato "case_" seguido de uno o más dígitos.

# Define una clase de datos inmutable y optimizada en memoria para representar la estructura de un caso del conjunto de datos KiTS23.
# Almacena el identificador textual completo del caso (ej. "case_00000").
# Almacena el número entero extraído del caso para facilitar ordenamientos y operaciones lógicas.
# Almacena la ruta absoluta al directorio físico que contiene los archivos del caso.
# Almacena la ruta al archivo NIfTI principal que contiene el volumen de imagen médica.
# Almacena la ruta al archivo NIfTI de la segmentación de referencia.
# Almacena opcionalmente la ruta al directorio de instancias si este existe, o None en caso contrario.

# Propiedad lógica que determina de forma rápida si el caso actual cuenta con un directorio de instancias disponible.

# Función para extraer y convertir a entero el número de caso a partir del nombre del directorio.
# Comprueba si el nombre del directorio cumple estrictamente con el patrón predefinido de casos.

# Lanza un error de valor si el nombre del directorio no coincide con la nomenclatura esperada del dataset.

# Convierte y devuelve la parte numérica extraída de la coincidencia como un número entero.

# Función para descubrir, validar y cargar toda la información y rutas asociadas a un caso individual de KiTS23.
# Resuelve la ruta del directorio del caso a su forma absoluta y canónica.

# Obtiene el número de caso entero llamando a la función de análisis correspondiente sobre el nombre del directorio.

# Define las rutas esperadas para el volumen de imagen, la máscara de segmentación y el directorio opcional de instancias.

# Verifica de forma estricta que el archivo de imagen médica requerido exista físicamente en el disco.

# Lanza un error de archivo no encontrado si falta el volumen principal de imagen.

# Verifica de forma estricta que el archivo de segmentación de referencia exista físicamente.

# Lanza un error de archivo no encontrado si falta la segmentación de referencia.

# Construye y devuelve una instancia estructurada del caso con sus respectivas rutas, validando la existencia condicional del subdirectorio de instancias.

# Función para explorar la raíz de un dataset KiTS23 y descubrir todos los casos válidos contenidos en ella.
# Resuelve la ruta raíz del conjunto de datos a su formato absoluto.

# Comprueba si el directorio raíz especificado para el dataset existe físicamente.

# Lanza una excepción si la ruta del dataset no existe.

# Comprueba que la ruta proporcionada corresponda efectivamente a un directorio y no a un archivo.

# Lanza un error si la ruta raíz no es un directorio válido.

# Busca y lista de forma eficiente todos los subdirectorios dentro de la raíz que coincidan formalmente con el patrón de nomenclatura de casos.

# Procesa la lista de candidatos encontrados utilizando la función de descubrimiento individual para cada ruta.

# Devuelve una tupla inmutable con todos los casos descubiertos, ordenados ascendentemente de forma estricta según su número de caso entero.