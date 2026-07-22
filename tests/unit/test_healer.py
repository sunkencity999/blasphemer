"""Unit tests for the DPO capability-healing module (non-training parts)."""

from types import SimpleNamespace

import pytest

from heretic.config import Settings
from heretic.healer import Healer, preference_columns_to_keep


# Settings has cli_parse_args=True; stop it from parsing pytest's argv.
@pytest.fixture(autouse=True)
def disable_cli_parsing(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["pytest"])


class TestPreferenceColumns:
    def test_keeps_only_dpo_columns_and_order(self):
        available = ["source", "chosen", "id", "rejected", "prompt", "score"]
        # Order follows the canonical prompt/chosen/rejected ordering.
        assert preference_columns_to_keep(available) == ["prompt", "chosen", "rejected"]

    def test_implicit_prompt_format(self):
        assert preference_columns_to_keep(["chosen", "rejected"]) == ["chosen", "rejected"]

    def test_missing_required_columns(self):
        # A non-preference dataset yields nothing usable.
        assert preference_columns_to_keep(["text", "label"]) == []


class TestHealerInit:
    def _tokenizer(self, pad=None):
        return SimpleNamespace(pad_token=pad, eos_token="<eos>")

    def test_sets_pad_token_from_eos_when_missing(self):
        tok = self._tokenizer(pad=None)
        Healer(model=SimpleNamespace(), tokenizer=tok, settings=Settings(model="m"))
        assert tok.pad_token == "<eos>"

    def test_preserves_existing_pad_token(self):
        tok = self._tokenizer(pad="<pad>")
        Healer(model=SimpleNamespace(), tokenizer=tok, settings=Settings(model="m"))
        assert tok.pad_token == "<pad>"

    def test_unwraps_compiled_model(self):
        real = SimpleNamespace(name="real")
        compiled = SimpleNamespace(_orig_mod=real)  # torch.compile() wrapper shape
        healer = Healer(model=compiled, tokenizer=self._tokenizer(), settings=Settings(model="m"))
        assert healer.model is real

    def test_uses_model_directly_when_not_compiled(self):
        real = SimpleNamespace(name="real")
        healer = Healer(model=real, tokenizer=self._tokenizer(), settings=Settings(model="m"))
        assert healer.model is real


class TestHealConfig:
    def test_defaults(self):
        s = Settings(model="m")
        assert s.heal is False
        assert s.heal_only is False
        assert s.heal_dataset == "mlabonne/orpo-dpo-mix-40k"
        assert s.heal_samples == 2000
        assert s.heal_epochs == 1
        assert 0 < s.heal_beta <= 1

    def test_overrides(self):
        s = Settings(model="m", heal_only=True, heal_samples=500, heal_dataset="foo/bar")
        assert s.heal_only is True
        assert s.heal_samples == 500
        assert s.heal_dataset == "foo/bar"
