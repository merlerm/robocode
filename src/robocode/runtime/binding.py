"""Environment endpoint metadata shared by development and evaluation."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EnvironmentBinding:
    """Connection details passed to an agent; the token is omitted from logs."""

    service_id: str
    host: str
    port: int
    token: str = field(repr=False)
    strict: bool = False
    artifact_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    def client_host(self, container_backend: str) -> str:
        """Docker uses the host gateway; Apptainer installs a loopback relay."""
        return {
            "docker": "host.docker.internal",
            "apptainer": self.host,
            "local": self.host,
        }[container_backend]
