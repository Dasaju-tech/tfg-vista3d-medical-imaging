# Importa la anotación de futuros para permitir referencias adelantadas de tipos en Python (PEP 563)
from __future__ import annotations

# Importa el decorador dataclass para crear clases orientadas a datos de forma concisa
from dataclasses import dataclass
# Importa Path para la manipulación orientada a objetos de rutas del sistema de archivos
from pathlib import Path


# Define una clase de datos inmutable (frozen=True) y optimizada en memoria (slots=True)
@dataclass(frozen=True, slots=True)
class ProjectPaths:
    # Atributo base que almacena la ruta raíz del proyecto
    root: Path

    # Propiedad que devuelve la ruta al directorio de datos dentro de la raíz
    @property
    def data(self) -> Path:
        return self.root / "data"

    # Propiedad que devuelve la ruta al directorio de datos crudos (raw)
    @property
    def raw(self) -> Path:
        return self.data / "raw"

    # Propiedad que devuelve la ruta al directorio de datos procesados
    @property
    def processed(self) -> Path:
        return self.data / "processed"

    # Propiedad que devuelve la ruta al directorio de salidas (outputs)
    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    # Propiedad que devuelve la ruta al directorio de manifiestos dentro de las salidas
    @property
    def manifests(self) -> Path:
        return self.outputs / "manifests"


# Función para descubrir automáticamente la raíz del proyecto buscando marcadores específicos
def discover_project_root(start: Path | None = None) -> Path:
    # Define el directorio de inicio como el proporcionado o el directorio de trabajo actual, resolviendo su ruta absoluta
    current = (start or Path.cwd()).resolve()

    # Recorre el directorio actual y cada uno de sus directorios padres hacia arriba
    for candidate in (current, *current.parents):
        # Comprueba si en el directorio candidato existe el archivo de configuración pyproject.toml
        if (candidate / "pyproject.toml").exists():
            return candidate
        # Comprueba si en el directorio candidato existe el directorio de control de versiones .git
        if (candidate / ".git").exists():
            return candidate

    # Lanza un error en tiempo de ejecución si no se encuentra ningún marcador de raíz válido en la jerarquía
    raise RuntimeError(
        "Could not locate the project root. "
        "Expected a parent directory containing pyproject.toml or .git."
    )