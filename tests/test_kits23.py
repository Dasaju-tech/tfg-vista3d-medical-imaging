from pathlib import Path

import pytest

from vista3d_tfg.data.kits23 import (
    discover_case,
    discover_cases,
    parse_case_number,
)


def create_fake_case(
    root: Path,
    case_name: str,
    *,
    instances: bool = False,
) -> Path:
    case_directory = root / case_name
    case_directory.mkdir()

    (case_directory / "imaging.nii.gz").touch()
    (case_directory / "segmentation.nii.gz").touch()

    if instances:
        (case_directory / "instances").mkdir()

    return case_directory


def test_parse_case_number() -> None:
    assert parse_case_number("case_00000") == 0
    assert parse_case_number("case_00142") == 142


@pytest.mark.parametrize(
    "name",
    [
        "case",
        "case_xyz",
        "patient_00001",
        "case-00001",
    ],
)
def test_parse_case_number_rejects_invalid_names(
    name: str,
) -> None:
    with pytest.raises(ValueError):
        parse_case_number(name)


def test_discover_case(tmp_path: Path) -> None:
    directory = create_fake_case(
        tmp_path,
        "case_00042",
        instances=True,
    )

    case = discover_case(directory)

    assert case.case_id == "case_00042"
    assert case.case_number == 42
    assert case.imaging_path.name == "imaging.nii.gz"
    assert case.segmentation_path.name == "segmentation.nii.gz"
    assert case.has_instances


def test_discover_cases_are_sorted(
    tmp_path: Path,
) -> None:
    create_fake_case(tmp_path, "case_00020")
    create_fake_case(tmp_path, "case_00003")
    create_fake_case(tmp_path, "case_00100")

    cases = discover_cases(tmp_path)

    assert [
        case.case_number
        for case in cases
    ] == [3, 20, 100]


def test_missing_segmentation_is_rejected(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "case_00001"
    directory.mkdir()

    (directory / "imaging.nii.gz").touch()

    with pytest.raises(FileNotFoundError):
        discover_case(directory)


# Importa la clase Path del módulo pathlib para la gestión de rutas de archivos.
# Importa la librería pytest para la creación y ejecución de pruebas unitarias.
# Importa las funciones de descubrimiento y análisis de casos desde el módulo de datos del proyecto KiTS23.

# Define una función auxiliar para crear un caso falso o simulado en el disco con fines de prueba.
# Crea el directorio correspondiente al caso dentro de la ruta raíz especificada.

# Crea archivos vacíos que simulan el volumen de imagen médica y la máscara de segmentación de referencia.

# Crea opcionalmente un subdirectorio de instancias si el parámetro correspondiente está activado.

# Devuelve la ruta completa del directorio del caso creado.

# Define una prueba unitaria para verificar que la función analiza correctamente los números de caso a partir de nombres válidos.
# Comprueba que el nombre formateado con ceros a la izquierda devuelva su equivalente entero correcto.
# Comprueba otro caso numérico diferente para asegurar robustez.

# Utiliza el decorador de pytest para parametrizar la prueba con múltiples nombres de directorios inválidos.
# Define una lista de cadenas de texto con formatos incorrectos que no deben ser aceptados.

# Define una prueba unitaria para verificar que se rechacen nombres de directorios inválidos.
# Utiliza un contexto de prueba para asegurar que se lanza una excepción de tipo ValueError.
# Ejecuta la función de análisis con el nombre inválido proporcionado por la parametrización.

# Define una prueba unitaria para verificar el descubrimiento correcto de un caso individual utilizando un directorio temporal.
# Crea un caso falso con un número específico y habilita la creación del directorio de instancias.

# Ejecuta la función de descubrimiento sobre el directorio simulado.

# Afirma que el identificador textual del caso coincida exactamente con el esperado.
# Afirma que el número entero del caso sea el correcto.
# Afirma que el nombre del archivo de imagen sea el archivo NIfTI esperado.
# Afirma que el nombre del archivo de segmentación sea el esperado.
# Afirma que la propiedad que comprueba la existencia de instancias sea verdadera.

# Define una prueba unitaria para verificar que la función de descubrimiento múltiple devuelva los casos ordenados.
# Crea múltiples casos falsos en orden desordenado dentro del directorio temporal.
# Crea el caso con número 20.
# Crea el caso con número 3.
# Crea el caso con número 100.

# Ejecuta la función de descubrimiento masivo sobre la ruta raíz temporal.

# Afirma que la lista de números de caso extraídos de los resultados esté ordenada estrictamente de menor a mayor.

# Define una prueba unitaria para verificar que la falta de un archivo de segmentación obligatorio sea rechazada adecuadamente.
# Crea manualmente un directorio de caso sin seguir el asistente completo.

# Crea únicamente el archivo de imagen médica simulado, omitiendo deliberadamente la segmentación.

# Utiliza un contexto de prueba para asegurar que se lance una excepción de archivo no encontrado al intentar descubrir el caso.