"""Environment-driven configuration for the API.

Everything the service needs to differ between a laptop and a deployment lives
here and is read from the environment. Nothing about a hostname, a port, a
filesystem path, or an allowed origin is written into the code.

**On secrets:** this service currently has none. It holds no database
credentials, no third-party API keys, and no signing material -- it loads two
model files from disk and answers HTTP. That is worth stating plainly rather
than inventing a secrets mechanism that guards nothing. If one is added later
(a database URL, an auth token), it belongs in this module, read from the
environment the same way, and must never acquire a real default value.

All variables are prefixed ``PERM_API_`` except ``PORT``, which is injected by
Railway, Render, Fly, and Cloud Run under that exact name.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEVELOPMENT = "development"
PRODUCTION = "production"

# Used only when PERM_API_ENV is development. A deployment must name its
# frontend explicitly; see Settings.validate().
DEV_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
)


class ConfigError(RuntimeError):
    """Raised when the environment describes a configuration that cannot be served."""


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(
        f"{name}={raw!r} is not a boolean. Use one of: 1/0, true/false, yes/no, on/off."
    )


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not an integer.") from exc


def _split_origins(raw: str) -> list[str]:
    """Split a comma-separated origin list, tolerating spaces and trailing slashes."""
    return [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]


@dataclass(frozen=True)
class Settings:
    environment: str = DEVELOPMENT
    cors_origins: list[str] = field(default_factory=list)
    artifact_dir: Path = PROJECT_ROOT / "src" / "models" / "artifacts"
    store_capacity: int = 500
    enable_docs: bool = True
    port: int = 8000

    @property
    def is_production(self) -> bool:
        return self.environment == PRODUCTION

    def validate(self) -> list[str]:
        """Fail on configurations that cannot be served; warn on risky ones.

        Returns advisory notes. Raises ConfigError for anything unserveable, at
        import time, so a misconfigured deployment dies immediately with a clear
        message instead of half-working under load.
        """
        notes: list[str] = []

        if self.environment not in {DEVELOPMENT, PRODUCTION}:
            raise ConfigError(
                f"PERM_API_ENV={self.environment!r} is not recognised. "
                f"Use {DEVELOPMENT!r} or {PRODUCTION!r}."
            )

        # A wildcard origin in production would let any site on the internet
        # drive this API from a visitor's browser. Refuse rather than warn:
        # "*" is almost always a debugging shortcut someone forgot to remove.
        if self.is_production and "*" in self.cors_origins:
            raise ConfigError(
                "PERM_API_CORS_ORIGINS contains '*', which is refused in production. "
                "List the deployed frontend's exact origin instead, e.g. "
                "PERM_API_CORS_ORIGINS=https://your-app.vercel.app"
            )

        for origin in self.cors_origins:
            if origin == "*":
                continue
            if not origin.startswith(("http://", "https://")):
                raise ConfigError(
                    f"CORS origin {origin!r} must include a scheme, e.g. "
                    f"https://{origin}. An origin is scheme + host + port, never a "
                    "bare domain and never a path."
                )
            if self.is_production and origin.startswith("http://") and (
                "localhost" not in origin and "127.0.0.1" not in origin
            ):
                notes.append(
                    f"origin {origin} uses plain http in production; prefer https"
                )

        if self.is_production and not self.cors_origins:
            # Legitimate and in fact the most secure arrangement: when the
            # frontend proxies through its own /api route, the browser never
            # makes a cross-origin request and no allowance is needed.
            notes.append(
                "no CORS origins set. Correct if the frontend calls this API "
                "server-side through its own /api proxy (API_INTERNAL_URL). If the "
                "browser is meant to call this API directly, set "
                "PERM_API_CORS_ORIGINS to the frontend's origin."
            )

        if self.store_capacity < 1:
            raise ConfigError(
                f"PERM_API_STORE_CAPACITY={self.store_capacity} must be at least 1; "
                "the explanation store needs room for one prediction."
            )

        if self.is_production and self.enable_docs:
            notes.append(
                "interactive docs are exposed at /docs. Fine for a public research "
                "project; set PERM_API_ENABLE_DOCS=false to hide them."
            )

        if not self.artifact_dir.exists():
            notes.append(
                f"artifact directory {self.artifact_dir} does not exist. The service "
                "will start but report itself not ready, and prediction endpoints "
                "will return 503. See DEPLOYMENT.md for how to supply the models."
            )

        return notes


def load_settings() -> Settings:
    """Build Settings from the process environment."""
    environment = _env("PERM_API_ENV", DEVELOPMENT).lower() or DEVELOPMENT

    raw_origins = _env("PERM_API_CORS_ORIGINS")
    if raw_origins:
        origins = _split_origins(raw_origins)
    elif environment == DEVELOPMENT:
        origins = list(DEV_CORS_ORIGINS)
    else:
        # No localhost fallback in production: a deployment that forgot to
        # configure origins should not silently inherit a developer's laptop.
        origins = []

    artifact_dir = _env("PERM_API_ARTIFACT_DIR")

    return Settings(
        environment=environment,
        cors_origins=origins,
        artifact_dir=Path(artifact_dir) if artifact_dir
        else PROJECT_ROOT / "src" / "models" / "artifacts",
        store_capacity=_env_int("PERM_API_STORE_CAPACITY", 500),
        enable_docs=_env_bool("PERM_API_ENABLE_DOCS", True),
        port=_env_int("PORT", 8000),
    )


settings = load_settings()
CONFIG_NOTES = settings.validate()
