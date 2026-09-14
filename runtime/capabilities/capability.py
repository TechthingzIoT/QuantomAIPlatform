"""QAIR capability model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Capability:
    """Describe a capability exposed by QAIR."""

    name: str
    available: bool
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("name must be a string.")

        name = self.name.strip()
        if not name:
            raise ValueError("name cannot be empty.")

        if not isinstance(self.available, bool):
            raise TypeError("available must be a boolean.")

        if not isinstance(self.description, str):
            raise TypeError("description must be a string.")

        description = self.description.strip()
        if not description:
            raise ValueError("description cannot be empty.")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)
