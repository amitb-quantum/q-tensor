from pathlib import Path

import pytest

from q_tensor.config import ConfigError, load_config


def test_smoke_config_is_valid() -> None:
    config = load_config(Path("configs/smoke.toml"))
    assert config["upstream"]["commit"] == "0569f848d9a3e385d6c20162c534a8031f6a69c5"


def test_missing_seed_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text('schema_version = 1\n[experiment]\nname="x"\ncategory="reproduced_result"\n')
    with pytest.raises(ConfigError, match="seed"):
        load_config(path)
