from pathlib import Path

import pytest

from vista3d_tfg.config import ProjectPaths, discover_project_root


def test_project_paths_are_derived_from_root(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)

    assert paths.data == tmp_path / "data"
    assert paths.raw == tmp_path / "data" / "raw"
    assert paths.processed == tmp_path / "data" / "processed"
    assert paths.outputs == tmp_path / "outputs"
    assert paths.manifests == tmp_path / "outputs" / "manifests"


def test_discover_project_root_from_nested_directory(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    nested = project_root / "src" / "package"
    nested.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text("[project]\nname='demo'\n")

    assert discover_project_root(nested) == project_root.resolve()


def test_discover_project_root_raises_without_marker(tmp_path: Path) -> None:
    isolated = tmp_path / "isolated"
    isolated.mkdir()

    with pytest.raises(RuntimeError, match="Could not locate the project root"):
        discover_project_root(isolated)
