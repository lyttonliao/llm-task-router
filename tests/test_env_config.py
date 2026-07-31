import os
from pathlib import Path
from unittest.mock import patch

from llm_task_router import env_config
from llm_task_router.env_config import load_dotenv_if_present


def test_missing_env_file_does_nothing(tmp_path):
    missing = tmp_path / ".env"

    with patch.dict(os.environ, {}, clear=False):
        load_dotenv_if_present(missing)  # must not raise


def test_loads_simple_key_value_pairs(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=bar\nBAZ=qux\n")
    env_without = {k: v for k, v in os.environ.items() if k not in ("FOO", "BAZ")}

    with patch.dict(os.environ, env_without, clear=True):
        load_dotenv_if_present(env_file)

        assert os.environ["FOO"] == "bar"
        assert os.environ["BAZ"] == "qux"


def test_existing_env_var_is_not_overridden(tmp_path):
    """The whole point of setdefault over []=: a real shell-exported value,
    or the launchd plists' own EnvironmentVariables block, must win over
    whatever .env says."""
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=from_file\n")

    with patch.dict(os.environ, {"FOO": "from_shell"}, clear=False):
        load_dotenv_if_present(env_file)

        assert os.environ["FOO"] == "from_shell"


def test_skips_blank_lines_and_comments(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("# a comment\n\nFOO=bar\n")
    env_without = {k: v for k, v in os.environ.items() if k != "FOO"}

    with patch.dict(os.environ, env_without, clear=True):
        load_dotenv_if_present(env_file)

        assert os.environ["FOO"] == "bar"
        assert "# a comment" not in os.environ


def test_strips_surrounding_quotes(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text('FOO="bar baz"\nQUX=\'single\'\n')
    env_without = {k: v for k, v in os.environ.items() if k not in ("FOO", "QUX")}

    with patch.dict(os.environ, env_without, clear=True):
        load_dotenv_if_present(env_file)

        assert os.environ["FOO"] == "bar baz"
        assert os.environ["QUX"] == "single"


def test_ignores_lines_without_an_equals_sign(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("not a valid line\nFOO=bar\n")
    env_without = {k: v for k, v in os.environ.items() if k != "FOO"}

    with patch.dict(os.environ, env_without, clear=True):
        load_dotenv_if_present(env_file)

        assert os.environ["FOO"] == "bar"


def test_default_env_file_resolves_next_to_the_package_not_cwd():
    """Regression guard for the actual design goal: this must be
    Path(__file__)-derived (repo root), not Path.cwd() - the whole reason
    for this module existing is that llm-chat is called from arbitrary
    working directories (see CLAUDE.md, "Calling these from any directory,
    on any machine"). This test file lives at tests/test_env_config.py, one
    directory below repo root, same depth as llm_task_router/env_config.py -
    both should resolve to the same repo_root/.env."""
    expected = Path(__file__).resolve().parent.parent / ".env"

    assert env_config._ENV_FILE == expected
