"""
=========================================================
QAIR Model Manager
=========================================================

High-level model lifecycle management.

Responsibilities
----------------
• Discover installed models
• Track the active model
• Validate models before activation
• Persist registry changes

Author:
    TIOTAIROBOTIX
=========================================================
"""
from runtime.models.discovery import discover_default_models
from runtime.models.exceptions import (
    InvalidModel,
    ModelNotFound,
)
from runtime.models.model import Model
from runtime.models.registry import (
    get_active_model,
    set_active_model,
)
from runtime.models.validator import validate_model


class ModelManager:
    """
    Central interface for QAIR model management.
    """

    def __init__(self):
        self._models: list[Model] = []

    # ==================================================
    # Discovery
    # ==================================================

    def refresh(self) -> list[Model]:
        """
        Rediscover installed models.
        """

        self._models = discover_default_models()
        return self._models

    def list_models(self) -> list[Model]:
        """
        Return every discovered model.
        """

        if not self._models:
            self.refresh()

        return self._models

    # ==================================================
    # Lookup
    # ==================================================

    def get(self, model_name: str) -> Model:
        """
        Retrieve a model by filename.

        Raises
        ------
        ModelNotFound
        """

        for model in self.list_models():

            if model.name == model_name:
                return model

        raise ModelNotFound(f"Model '{model_name}' not found.")

    def exists(self, model_name: str) -> bool:
        """
        Check whether a model exists.
        """

        try:
            self.get(model_name)
            return True

        except ModelNotFound:
            return False

    # ==================================================
    # Active Model
    # ==================================================

    def active_model(self) -> Model | None:
        """
        Return the currently active model.
        """

        active = get_active_model()

        if active is None:
            return None

        try:
            return self.get(active)

        except ModelNotFound:
            return None

    def activate(self, model_name: str) -> Model:
        """
        Validate and activate a model.
        """

        model = self.get(model_name)

        valid, reason = validate_model(model.path)

        if not valid:
            raise InvalidModel(reason)

        set_active_model(model.name)

        return model

    # Backwards compatibility
    set_current_model = activate

    # ==================================================
    # Statistics
    # ==================================================

    def count(self) -> int:
        """
        Number of discovered models.
        """

        return len(self.list_models())

    def total_size(self) -> int:
        """
        Total size of all discovered models.
        """

        return sum(model.size for model in self.list_models())

    def summary(self) -> dict:
        """
        Runtime summary.
        """

        active = self.active_model()

        return {
            "installed_models": self.count(),
            "active_model": active.name if active else None,
            "total_size": self.total_size(),
        }

    # ==================================================
    # Convenience
    # ==================================================

    def __len__(self) -> int:
        return self.count()

    def __iter__(self):
        return iter(self.list_models())

    def __contains__(self, model_name: str) -> bool:
        return self.exists(model_name)