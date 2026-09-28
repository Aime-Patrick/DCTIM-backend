from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.dependencies as deps
from app.main import create_app
from app.modules.auth.users import get_user_store
from app.core.config import Settings
from app.modules.rag.infrastructure.storage import (
    CloudinaryStorage,
    LocalFileStorage,
    StorageError,
    StoredFile,
    build_storage,
)


class FakeCloudinaryClient:
    """Minimal stand-in for ``cloudinary.Cloudinary``."""

    def __init__(self, result: dict | None = None, raises: Exception | None = None) -> None:
        self._result = result or {
            "secure_url": "https://res.cloudinary.com/demo/raw/upload/v1/uploads/ws-1/abc_report.pdf",
            "public_id": "uploads/ws-1/abc_report",
            "bytes": 4,
        }
        self._raises = raises
        self.uploads: list[dict] = []
        self.destroys: list[tuple[str, str]] = []

    def upload(self, data: bytes, **options):
        if self._raises is not None:
            raise self._raises
        self.uploads.append({"data": data, **options})
        return self._result

    def destroy(self, public_id: str, **options):
        if self._raises is not None:
            raise self._raises
        self.destroys.append((public_id, options.get("resource_type", "")))
        return {"result": "ok"}


# ---------------------------------------------------------------------------
# Local backend
# ---------------------------------------------------------------------------

def test_local_storage_writes_under_workspace_directory(tmp_path: Path) -> None:
    store = LocalFileStorage(tmp_path)

    stored = store.save("ws-1", "abc123", "Annual Report.pdf", b"%PDF-1.4")

    assert stored.provider == "local"
    assert stored.reference == "ws-1/abc123_Annual_Report.pdf"
    assert stored.url is None
    assert (tmp_path / stored.reference).read_bytes() == b"%PDF-1.4"


def test_local_storage_strips_path_traversal_from_filename(tmp_path: Path) -> None:
    store = LocalFileStorage(tmp_path)

    stored = store.save("../../etc", "abc123", "../../etc/passwd", b"data")

    assert stored.reference == "etc/abc123_passwd"
    assert (tmp_path / stored.reference).exists()
    assert not (tmp_path.parent / "etc").exists()


def test_local_storage_delete_is_idempotent(tmp_path: Path) -> None:
    store = LocalFileStorage(tmp_path)
    stored = store.save("ws-1", "abc123", "notes.txt", b"hello")

    store.delete(stored.reference)
    store.delete(stored.reference)

    assert not (tmp_path / stored.reference).exists()


# ---------------------------------------------------------------------------
# Cloudinary backend
# ---------------------------------------------------------------------------

def test_cloudinary_storage_uploads_as_raw_and_returns_secure_url() -> None:
    client = FakeCloudinaryClient()
    store = CloudinaryStorage("demo", "key", "secret", folder="dc-tim/uploads", client=client)

    stored = store.save("workspace one", "abc123", "Annual Report.pdf", b"%PDF-1.4")

    assert stored.provider == "cloudinary"
    assert stored.reference.startswith("https://res.cloudinary.com/")
    assert stored.url == stored.reference
    assert stored.public_id == "uploads/ws-1/abc_report"

    options = client.uploads[0]
    assert options["resource_type"] == "raw"
    assert options["public_id"] == "dc-tim/uploads/workspace-one/abc123_Annual_Report"
    assert options["format"] == "pdf"
    assert options["data"] == b"%PDF-1.4"
    assert options["timeout"] == 60.0


def test_cloudinary_storage_omits_format_when_extension_is_unsafe() -> None:
    client = FakeCloudinaryClient()
    store = CloudinaryStorage("demo", "key", "secret", client=client)

    store.save("ws-1", "abc123", "report.verylongextension", b"data")

    assert "format" not in client.uploads[0]


def test_cloudinary_storage_raises_storage_error_on_upload_failure() -> None:
    client = FakeCloudinaryClient(raises=RuntimeError("connection reset"))
    store = CloudinaryStorage("demo", "key", "secret", client=client)

    with pytest.raises(StorageError, match="Cloudinary upload failed"):
        store.save("ws-1", "abc123", "notes.txt", b"hello")


def test_cloudinary_storage_raises_storage_error_without_secure_url() -> None:
    client = FakeCloudinaryClient(result={"public_id": "uploads/ws-1/abc"})
    store = CloudinaryStorage("demo", "key", "secret", client=client)

    with pytest.raises(StorageError, match="no secure_url"):
        store.save("ws-1", "abc123", "notes.txt", b"hello")


def test_cloudinary_delete_recovers_public_id_from_url() -> None:
    client = FakeCloudinaryClient()
    store = CloudinaryStorage("demo", "key", "secret", client=client)
    reference = "https://res.cloudinary.com/demo/raw/upload/v1730000000/dc-tim/uploads/ws-1/abc_report.pdf"

    store.delete(reference)

    assert client.destroys == [("dc-tim/uploads/ws-1/abc_report", "raw")]


# ---------------------------------------------------------------------------
# Configuration / factory
# ---------------------------------------------------------------------------

def _settings(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    for name in (
        "RAG_ENVIRONMENT",
        "RAG_STORAGE_PROVIDER",
        "CLOUDINARY_CLOUD_NAME",
        "CLOUDINARY_API_KEY",
        "CLOUDINARY_API_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in overrides.items():
        monkeypatch.setenv(name, value)
    return Settings.from_environment()


def test_storage_defaults_to_local_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)

    assert settings.storage_provider == "local"
    assert isinstance(build_storage(settings), LocalFileStorage)


def test_storage_defaults_to_cloudinary_when_credentials_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(
        monkeypatch,
        CLOUDINARY_CLOUD_NAME="demo",
        CLOUDINARY_API_KEY="123",
        CLOUDINARY_API_SECRET="shh",
    )

    assert settings.storage_provider == "cloudinary"
    assert isinstance(build_storage(settings), CloudinaryStorage)


def test_storage_honours_explicit_local_override(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(
        monkeypatch,
        RAG_STORAGE_PROVIDER="local",
        CLOUDINARY_CLOUD_NAME="demo",
        CLOUDINARY_API_KEY="123",
        CLOUDINARY_API_SECRET="shh",
    )

    assert isinstance(build_storage(settings), LocalFileStorage)


def test_incomplete_cloudinary_credentials_fall_back_to_local_in_dev(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch, RAG_STORAGE_PROVIDER="cloudinary", CLOUDINARY_CLOUD_NAME="demo")

    assert settings.storage_provider == "local"


def test_incomplete_cloudinary_credentials_fail_loudly_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="CLOUDINARY_API_SECRET"):
        _settings(
            monkeypatch,
            RAG_ENVIRONMENT="production",
            RAG_STORAGE_PROVIDER="cloudinary",
            CLOUDINARY_CLOUD_NAME="demo",
            AUTH_JWT_SECRET="a-long-random-secret",
        )


# ---------------------------------------------------------------------------
# /rag/upload endpoint wiring
# ---------------------------------------------------------------------------

class RecordingStore:
    """In-memory BlobStore that records saves and deletes."""

    provider = "cloudinary"

    def __init__(self) -> None:
        self.saved: list[tuple[str, str, str, bytes]] = []
        self.deleted: list[str] = []

    def save(self, workspace_id: str, document_id: str, filename: str, data: bytes) -> StoredFile:
        self.saved.append((workspace_id, document_id, filename, data))
        return StoredFile(
            reference=f"https://res.cloudinary.com/demo/raw/upload/v1/uploads/{document_id}_{filename}",
            name=filename,
            size_bytes=len(data),
            provider=self.provider,
            url=f"https://res.cloudinary.com/demo/raw/upload/v1/uploads/{document_id}_{filename}",
            public_id=f"uploads/{document_id}",
        )

    def delete(self, reference: str) -> None:
        self.deleted.append(reference)


def _upload_client(monkeypatch: pytest.MonkeyPatch, store: RecordingStore) -> tuple[TestClient, dict]:
    get_user_store.cache_clear()
    deps.get_settings.cache_clear()
    deps.get_auth_service.cache_clear()
    deps._memory_vector_store = None
    deps._engine = None
    deps._session_factory = None
    os.environ.pop("DATABASE_URL", None)
    os.environ["RAG_EMBEDDING_PROVIDER"] = "hash"
    os.environ["RAG_ANSWER_PROVIDER"] = "demo"
    monkeypatch.setattr("app.modules.rag.api.build_storage", lambda settings: store)

    app = create_app()
    client = TestClient(app)
    login = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@dc-tim.ai", "password": "admin123"},
    )
    assert login.status_code == 200
    token = login.json()["access_token"]
    return client, {"Authorization": f"Bearer {token}"}


def test_upload_stores_via_configured_backend_and_returns_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RecordingStore()
    client, headers = _upload_client(monkeypatch, store)

    response = client.post(
        "/api/v1/rag/upload",
        headers=headers,
        files={"file": ("policy.md", b"# Policy\nTeacher training matters.", "text/markdown")},
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["file_path"].endswith(f"{body['document_id']}_policy.md")
    assert body["file_path"].startswith("https://res.cloudinary.com/")
    assert body["file_url"] == body["file_path"]
    # The blob was written with the same id the document row received.
    assert store.saved[0][1] == body["document_id"]


def test_upload_rolls_back_the_blob_when_ingest_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RecordingStore()
    client, headers = _upload_client(monkeypatch, store)

    class _ExplodingService:
        def ingest(self, *args, **kwargs):
            raise ValueError("title and content are required")

    client.app.dependency_overrides[deps.get_rag_service] = lambda: _ExplodingService()
    response = client.post(
        "/api/v1/rag/upload",
        headers=headers,
        files={"file": ("policy.md", b"# Policy\nTeacher training matters.", "text/markdown")},
    )

    assert response.status_code == 422
    assert store.deleted, "the orphaned upload should be removed from storage"
