"""Blob storage for uploaded documents.

Two interchangeable backends implement the same :class:`BlobStore` contract so
the API layer never knows where a file ended up:

``LocalFileStorage``
    Writes to ``uploads/`` under the backend root.  Fine for local development
    and single-node deployments with a mounted volume, but **not** for
    serverless runtimes: the filesystem is ephemeral, so uploaded documents
    disappear on the next cold start.

``CloudinaryStorage``
    Uploads from the API process using the signed server-side SDK, so the
    ``CLOUDINARY_API_SECRET`` never leaves the backend.  The returned HTTPS URL
    is what gets persisted in ``rag_documents.file_path``.

Selection is driven by ``RAG_STORAGE_PROVIDER`` (see
``app.core.config.Settings``); when it is left blank Cloudinary is used
automatically if all three credentials are present, otherwise local disk.

Local directory layout
----------------------
uploads/
  <workspace_id>/
    <document_id>_<original_filename>

Rules
-----
- Filenames are sanitised to strip path traversal characters.
- The workspace sub-directory is created on demand.
- Overwrites are allowed (idempotent re-ingestion of the same file).
- The storage layer never reads the file content — that is the extractor's job.
- Documents are stored as Cloudinary ``raw`` resources so PDFs and Office
  files pass through untouched instead of being treated as images.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "BlobStore",
    "CloudinaryStorage",
    "LocalFileStorage",
    "StorageError",
    "StoredFile",
    "build_storage",
]


class StorageError(RuntimeError):
    """Raised when the storage backend cannot persist or remove an object."""


@dataclass(frozen=True)
class StoredFile:
    """The outcome of a single successful upload.

    Attributes
    ----------
    reference:
        Opaque, stable string persisted in ``rag_documents.file_path``.  For the
        local backend it is a storage-relative path; for Cloudinary it is the
        public HTTPS URL.
    url:
        Directly fetchable URL, or ``None`` for backends that cannot serve the
        object (local disk).
    public_id:
        Backend-native identifier used for deletions, or ``None`` when the
        backend has none.
    """

    reference: str
    name: str
    size_bytes: int
    provider: str
    url: str | None = None
    public_id: str | None = None


class BlobStore(Protocol):
    """Contract for persisting the raw bytes of an uploaded document."""

    @property
    def provider(self) -> str: ...

    def save(
        self,
        workspace_id: str,
        document_id: str,
        filename: str,
        data: bytes,
    ) -> StoredFile: ...

    def delete(self, reference: str) -> None: ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _default_upload_root() -> Path:
    """Return the upload root, resolved relative to the backend package root."""
    import os

    env_val = os.environ.get("RAG_UPLOAD_DIR", "").strip()
    if env_val:
        return Path(env_val).resolve()
    # backend/app/modules/rag/infrastructure/storage.py → go up 5 levels
    return Path(__file__).parents[4] / "uploads"


def _sanitise_filename(name: str) -> str:
    """Strip path separators and control chars; keep extension."""
    # Replace anything that isn't alphanumeric, dot, dash, or underscore.
    safe = re.sub(r"[^\w.\-]", "_", Path(name).name)
    # Collapse consecutive underscores produced by the substitution.
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "unnamed"


def _sanitise_segment(value: str, fallback: str = "unnamed") -> str:
    """Sanitise a single path/folder segment (no dots, no slashes)."""
    safe = re.sub(r"[^\w\-]", "-", str(value or "").strip())
    safe = re.sub(r"-{2,}", "-", safe).strip("-_")
    return safe or fallback


def _sanitise_folder(value: str, fallback: str = "uploads") -> str:
    """Sanitise a slash-separated folder path, keeping the separators."""
    segments = [_sanitise_segment(part) for part in str(value or "").split("/")]
    return "/".join(part for part in segments if part) or fallback


# ---------------------------------------------------------------------------
# Local filesystem backend
# ---------------------------------------------------------------------------

class LocalFileStorage:
    """Handles saving uploaded files to the local filesystem.

    Parameters
    ----------
    upload_root:
        Base directory for all uploads.  Defaults to ``uploads/`` next to
        the ``backend/`` folder.  Pass an explicit path in tests.
    """

    def __init__(self, upload_root: Path | None = None) -> None:
        self._root = upload_root or _default_upload_root()

    @property
    def provider(self) -> str:
        return "local"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        workspace_id: str,
        document_id: str,
        filename: str,
        data: bytes,
    ) -> StoredFile:
        """Write *data* to disk and return a :class:`StoredFile`.

        Creates the workspace sub-directory if it does not exist.
        """
        try:
            workspace_dir = self._root / _sanitise_segment(workspace_id)
            workspace_dir.mkdir(parents=True, exist_ok=True)

            safe_name = f"{document_id}_{_sanitise_filename(filename)}"
            dest = workspace_dir / safe_name
            dest.write_bytes(data)
        except OSError as exc:
            raise StorageError(f"Could not write {filename!r} to local storage: {exc}") from exc

        return StoredFile(
            reference=self.relative(dest),
            name=filename,
            size_bytes=len(data),
            provider=self.provider,
        )

    def delete(self, reference: str) -> None:
        """Remove a stored object.  Missing files are ignored."""
        try:
            self.resolve(reference).unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - defensive
            raise StorageError(f"Could not delete {reference!r}: {exc}") from exc

    def resolve(self, relative_path: str) -> Path:
        """Return the absolute path for a stored *relative_path*.

        The relative path is stored in ``rag_documents.file_path`` as
        ``<workspace_id>/<document_id>_<filename>``.
        """
        return self._root / relative_path

    def relative(self, absolute_path: Path) -> str:
        """Convert an absolute path back to a storage-relative string."""
        return absolute_path.relative_to(self._root).as_posix()


# ---------------------------------------------------------------------------
# Cloudinary backend
# ---------------------------------------------------------------------------

class _SignedCloudinaryClient:
    """Thin wrapper over ``cloudinary.uploader`` bound to one account.

    The Python SDK has no per-account client object — it reads credentials from
    either the process-wide ``cloudinary.config()`` or from the call options.
    This wrapper takes the second route so concurrent requests cannot leak
    credentials into each other, and so the module never mutates global state
    at import time.
    """

    def __init__(self, cloud_name: str, api_key: str, api_secret: str) -> None:
        self._credentials = {
            "cloud_name": cloud_name,
            "api_key": api_key,
            "api_secret": api_secret,
            "secure": True,
        }

    def upload(self, data: bytes, **options: Any) -> dict:
        from cloudinary import uploader

        return uploader.upload(data, **self._credentials, **options)

    def destroy(self, public_id: str, **options: Any) -> dict:
        from cloudinary import uploader

        return uploader.destroy(public_id, **self._credentials, **options)


class CloudinaryStorage:
    """Uploads documents to Cloudinary from the API process.

    Parameters
    ----------
    cloud_name, api_key, api_secret:
        Cloudinary account credentials.  Uploads are signed server-side, so
        these stay on the backend and are never exposed to the browser.
    folder:
        Root folder inside the Cloudinary media library.
    resource_type:
        ``raw`` (default) keeps PDFs/DOCX/XLSX byte-identical.  Only switch to
        ``auto``/``image`` if the account is configured for image delivery.
    timeout_seconds:
        Upload timeout.  Keep it below the serverless function timeout so the
        platform can report a clean error instead of a hard kill.
    client:
        Pre-built client.  Injected by tests; built from the credentials above
        otherwise.
    """

    def __init__(
        self,
        cloud_name: str,
        api_key: str,
        api_secret: str,
        folder: str = "dc-tim/uploads",
        resource_type: str = "raw",
        timeout_seconds: float = 60.0,
        client: Any | None = None,
    ) -> None:
        self._cloud_name = cloud_name
        self._api_key = api_key
        self._api_secret = api_secret
        self._folder = _sanitise_folder(folder)
        self._resource_type = resource_type
        self._timeout_seconds = timeout_seconds
        self._client = client

    @property
    def provider(self) -> str:
        return "cloudinary"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        workspace_id: str,
        document_id: str,
        filename: str,
        data: bytes,
    ) -> StoredFile:
        """Upload *data* and return the resulting :class:`StoredFile`."""
        safe_name = _sanitise_filename(filename)
        public_id = (
            f"{self._folder}/"
            f"{_sanitise_segment(workspace_id)}/"
            f"{_sanitise_segment(document_id)}_{Path(safe_name).stem}"
        )

        options: dict[str, Any] = {
            "public_id": public_id,
            "resource_type": self._resource_type,
            "overwrite": True,
            "use_filename": False,
            "unique_filename": False,
            "invalidate": True,
            "tags": [_sanitise_segment(workspace_id)],
            "timeout": self._timeout_seconds,
        }
        # Raw uploads do not keep the extension in the delivery URL unless the
        # format is stated, which would leave the URL ending in a bare name.
        extension = Path(safe_name).suffix.lstrip(".").lower()
        if re.fullmatch(r"[a-z0-9]{1,10}", extension):
            options["format"] = extension

        try:
            result = self._get_client().upload(data, **options)
        except StorageError:
            raise
        except Exception as exc:  # cloudinary.Error and transport failures
            raise StorageError(f"Cloudinary upload failed for {filename!r}: {exc}") from exc

        if not isinstance(result, dict) or not result.get("secure_url"):
            raise StorageError(f"Cloudinary returned no secure_url for {filename!r}")

        return StoredFile(
            reference=str(result["secure_url"]),
            name=filename,
            size_bytes=int(result.get("bytes", len(data))),
            provider=self.provider,
            url=str(result["secure_url"]),
            public_id=str(result.get("public_id") or public_id),
        )

    def delete(self, reference: str) -> None:
        """Destroy an uploaded asset.  Missing assets are ignored."""
        public_id = self.public_id_from_reference(reference)
        if public_id is None:
            raise StorageError(f"Cannot derive a Cloudinary public_id from {reference!r}")
        try:
            self._get_client().destroy(
                public_id,
                resource_type=self._resource_type,
                invalidate=True,
            )
        except Exception as exc:
            raise StorageError(f"Cloudinary delete failed for {public_id!r}: {exc}") from exc

    @staticmethod
    def public_id_from_reference(reference: str) -> str | None:
        """Recover the asset public_id from a stored HTTPS delivery URL.

        Cloudinary URLs look like
        ``https://res.cloudinary.com/<cloud>/raw/upload/v<version>/<public_id>``.
        """
        match = re.search(
            r"/(?:image|raw|video|auto)/upload/(?:v\d+/)?(.+?)(?:\.[A-Za-z0-9]+)?$",
            reference or "",
        )
        return match.group(1) if match else None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import cloudinary  # noqa: F401
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise StorageError(
                "The 'cloudinary' package is required for RAG_STORAGE_PROVIDER=cloudinary. "
                "Install backend dependencies with `pip install -e .`."
            ) from exc

        self._client = _SignedCloudinaryClient(
            cloud_name=self._cloud_name,
            api_key=self._api_key,
            api_secret=self._api_secret,
        )
        return self._client


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_storage(settings: Any) -> BlobStore:
    """Return the blob store selected by *settings*.

    Raises
    ------
    StorageError
        If Cloudinary is selected but its credentials are incomplete.
    """
    if settings.storage_provider == "cloudinary":
        credentials = {
            "CLOUDINARY_CLOUD_NAME": settings.cloudinary_cloud_name,
            "CLOUDINARY_API_KEY": settings.cloudinary_api_key,
            "CLOUDINARY_API_SECRET": settings.cloudinary_api_secret,
        }
        missing = [name for name, value in credentials.items() if not value]
        if missing:
            raise StorageError(
                "Cloudinary storage is selected but these settings are missing: "
                + ", ".join(missing)
            )
        return CloudinaryStorage(
            cloud_name=settings.cloudinary_cloud_name or "",
            api_key=settings.cloudinary_api_key or "",
            api_secret=settings.cloudinary_api_secret or "",
            folder=settings.cloudinary_folder,
            timeout_seconds=settings.cloudinary_timeout_seconds,
        )

    upload_root = Path(settings.upload_dir).resolve() if settings.upload_dir else None
    return LocalFileStorage(upload_root)
