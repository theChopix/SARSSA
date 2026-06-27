"""Compute the options for parameter dropdowns that depend on another param.

Some parameters offer a menu whose contents change with another
parameter's current value — e.g. the embedding model list depends on
the chosen embedding provider. This module owns that mapping: a
resolver key (declared on a ``DependentDropdownHint``) is paired with
a function that turns the controlling value into the allowed values.
The plugin API layer dispatches here when serving such a dropdown.
"""

from collections.abc import Callable

from utils.embedder.registry import known_models as embedder_known_models

#: Maps a ``DependentDropdownHint.resolver`` key to a function turning
#: the controlling param's value into the dependent param's values.
CHOICE_RESOLVERS: dict[str, Callable[[str], list[str]]] = {
    "embedder_models": embedder_known_models,
}


def resolve_dependent_choices(resolver_key: str, value: str | None) -> list[dict[str, str]]:
    """Resolve a dependent dropdown's choices for a controlling value.

    Args:
        resolver_key: Key into :data:`CHOICE_RESOLVERS`.
        value: Current value of the controlling parameter.

    Returns:
        list[dict[str, str]]: ``{"label", "value"}`` options. Empty
            when *value* is falsy or not recognised by the resolver.

    Raises:
        KeyError: If *resolver_key* is not registered.
    """
    resolver = CHOICE_RESOLVERS[resolver_key]
    if not value:
        return []
    try:
        values = resolver(value)
    except ValueError:
        return []
    return [{"label": v, "value": v} for v in values]
