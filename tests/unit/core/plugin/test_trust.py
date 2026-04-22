"""Tests for plugin trust records."""

import json
from pathlib import Path

from ggshield.core.plugin.trust import PluginTrustStore, compute_file_sha256


class TestPluginTrustStore:
    """Tests for PluginTrustStore."""

    def test_trust_plugin_records_exact_hash(self, tmp_path: Path) -> None:
        store = PluginTrustStore(path=tmp_path / "plugin_trust.json")

        store.trust_plugin("testplugin", "abc123", "missing")

        assert store.is_trusted("testplugin", "abc123") is True
        assert store.is_trusted("testplugin", "def456") is False

        record = store.get_record("testplugin")
        assert record is not None
        assert record.sha256 == "abc123"
        assert record.signature_status == "missing"

    def test_revoke_plugin_removes_empty_store(self, tmp_path: Path) -> None:
        trust_store_path = tmp_path / "plugin_trust.json"
        store = PluginTrustStore(path=trust_store_path)
        store.trust_plugin("testplugin", "abc123", "missing")

        store.revoke_plugin("testplugin")

        assert trust_store_path.exists() is False
        assert store.get_record("testplugin") is None

    def test_invalid_store_returns_default_data(self, tmp_path: Path) -> None:
        trust_store_path = tmp_path / "plugin_trust.json"
        trust_store_path.write_text("not-json")
        store = PluginTrustStore(path=trust_store_path)

        assert store.get_record("testplugin") is None
        assert store.is_trusted("testplugin", "abc123") is False

    def test_compute_file_sha256(self, tmp_path: Path) -> None:
        file_path = tmp_path / "plugin.whl"
        file_path.write_bytes(b"wheel-bytes")

        digest = compute_file_sha256(file_path)

        assert len(digest) == 64
        assert digest == compute_file_sha256(file_path)

    def test_store_file_format(self, tmp_path: Path) -> None:
        trust_store_path = tmp_path / "plugin_trust.json"
        store = PluginTrustStore(path=trust_store_path)

        store.trust_plugin("testplugin", "abc123", "invalid")

        data = json.loads(trust_store_path.read_text())
        assert data["version"] == 1
        assert data["plugins"]["testplugin"]["sha256"] == "abc123"
        assert data["plugins"]["testplugin"]["signature_status"] == "invalid"
