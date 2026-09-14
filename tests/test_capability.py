import pytest

from runtime.capabilities import Capability


def test_capability_stores_values():
    capability = Capability(
        name="inference.generate",
        available=True,
        description="Generate responses.",
    )

    assert capability.name == "inference.generate"
    assert capability.available is True
    assert capability.description == "Generate responses."


def test_capability_normalizes_name_and_description():
    capability = Capability(
        name="  inference.generate  ",
        available=True,
        description="  Generate responses.  ",
    )

    assert capability.name == "inference.generate"
    assert capability.description == "Generate responses."


def test_capability_is_immutable():
    capability = Capability(
        name="inference.generate",
        available=True,
        description="Generate responses.",
    )

    with pytest.raises(AttributeError):
        capability.name = "other"


def test_capability_rejects_non_string_name():
    with pytest.raises(TypeError, match="name must be a string"):
        Capability(
            name=123,
            available=True,
            description="Generate responses.",
        )


def test_capability_rejects_empty_name():
    with pytest.raises(ValueError, match="name cannot be empty"):
        Capability(
            name="   ",
            available=True,
            description="Generate responses.",
        )


def test_capability_rejects_non_boolean_availability():
    with pytest.raises(TypeError, match="available must be a boolean"):
        Capability(
            name="inference.generate",
            available=1,
            description="Generate responses.",
        )


def test_capability_rejects_non_string_description():
    with pytest.raises(TypeError, match="description must be a string"):
        Capability(
            name="inference.generate",
            available=True,
            description=123,
        )


def test_capability_rejects_empty_description():
    with pytest.raises(
        ValueError,
        match="description cannot be empty",
    ):
        Capability(
            name="inference.generate",
            available=True,
            description="   ",
        )
