"""Base class for *compare* plugins.

A compare plugin runs analytical logic against both the current
pipeline context and a past pipeline run, then surfaces a side-by-
side display.  Lives in its own module to keep
:mod:`plugins.plugin_interface` focused on the core plugin
contract.
"""

import inspect
from dataclasses import replace
from typing import Any

from plugins.plugin_interface import (
    ArtifactSpec,
    BasePlugin,
    MissingContextError,
    PastRunsDropdownHint,
)
from utils.mlflow_manager import MLflowRunLoader


class BaseComparePlugin(BasePlugin):
    """Base class for compare plugins.

    Subclasses inherit past-run plumbing for free:

    - Auto-injects a :class:`PastRunsDropdownHint` for the
      ``past_run_id`` parameter using the subclass's
      :attr:`past_run_required_steps`.
    - Wraps the subclass's ``run()`` method to load the past run's
      ``context.json`` into ``self.past_context`` before the body
      executes; ``past_run_id`` must be passed as a keyword.
    - Exposes :meth:`load_past_artifact` for symmetric access to
      artifacts produced by steps of the past run.

    Subclasses must:

    1. Set the ``past_run_required_steps`` class attribute to the
       step keys an eligible past run must contain.
    2. Declare ``past_run_id: str`` as a parameter on ``run()``;
       additional parameters follow the usual conventions.

    Attributes:
        past_run_required_steps: Step keys the past run must have
            completed.  Forwarded to the auto-injected
            :class:`PastRunsDropdownHint`.
    """

    past_run_required_steps: list[str] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Wire past-run hint injection and ``run()`` wrapping.

        Args:
            **kwargs: Forwarded to :meth:`object.__init_subclass__`.
        """
        super().__init_subclass__(**kwargs)
        cls._merge_past_run_hint()
        cls._wrap_run_for_past_context()

    @classmethod
    def _merge_past_run_hint(cls) -> None:
        """Inject a :class:`PastRunsDropdownHint` for ``past_run_id``.

        Builds a fresh ``io_spec`` with the new hint appended to the
        existing ``param_ui_hints`` list, leaving any inherited spec
        on parent classes untouched.  Skips injection when an
        equivalent hint is already present.
        """
        existing = list(cls.io_spec.param_ui_hints)
        already_present = any(
            isinstance(h, PastRunsDropdownHint) and h.param_name == "past_run_id" for h in existing
        )
        if already_present:
            return
        existing.append(
            PastRunsDropdownHint(
                param_name="past_run_id",
                required_steps=list(cls.past_run_required_steps),
            )
        )
        cls.io_spec = replace(cls.io_spec, param_ui_hints=existing)

    @classmethod
    def _wrap_run_for_past_context(cls) -> None:
        """Wrap the subclass's ``run()`` to populate ``self.past_context``.

        Skipped when ``cls.run`` is still abstract (an intermediate
        base class) or already wrapped (multiple inheritance).
        """
        original = cls.run
        if getattr(original, "__isabstractmethod__", False):
            return
        if getattr(original, "__compare_wrapped__", False):
            return

        original_signature = inspect.signature(original)

        def wrapped(self: "BaseComparePlugin", *args: Any, **kwargs: Any) -> Any:
            """Load past_context, then delegate to the original run().

            Args:
                self: The plugin instance.
                *args: Positional arguments forwarded to ``run()``.
                **kwargs: Keyword arguments forwarded to ``run()``;
                    must include ``past_run_id``.

            Returns:
                Any: The original ``run()``'s return value.

            Raises:
                MissingContextError: If ``past_run_id`` is missing or
                    the past context cannot be loaded.
            """
            past_run_id = kwargs.get("past_run_id")
            if past_run_id is None:
                raise MissingContextError(
                    "Compare plugins require 'past_run_id' as a keyword argument"
                )
            self._load_past_context(past_run_id)
            return original(self, *args, **kwargs)

        wrapped.__compare_wrapped__ = True  # type: ignore[attr-defined]
        wrapped.__wrapped__ = original  # type: ignore[attr-defined]
        wrapped.__signature__ = original_signature  # type: ignore[attr-defined]
        wrapped.__doc__ = original.__doc__
        wrapped.__name__ = original.__name__
        wrapped.__qualname__ = original.__qualname__
        cls.run = wrapped  # type: ignore[method-assign]

    def _load_past_context(self, past_run_id: str) -> None:
        """Load the past run's ``context.json`` into ``self.past_context``.

        Args:
            past_run_id: MLflow run id of a parent pipeline run.

        Raises:
            MissingContextError: If the run has no ``context.json`` or
                MLflow cannot retrieve it.
        """
        loader = MLflowRunLoader(past_run_id)
        try:
            self.past_context = loader.get_json_artifact("context.json")
        except Exception as exc:
            raise MissingContextError(
                f"Failed to load context.json for past run '{past_run_id}': {exc}"
            ) from exc

    def load_past_artifact(
        self,
        step: str,
        filename: str,
        loader: str = "json",
        **loader_kwargs: Any,
    ) -> Any:
        """Load an artifact produced by *step* in the past run.

        Resolves the per-step run id through ``self.past_context``,
        constructs an :class:`ArtifactSpec`, and delegates to
        :meth:`BasePlugin._load_artifact` so every loader strategy
        the base class supports works the same here.

        Args:
            step: Step key to look up in ``self.past_context``.
            filename: Artifact filename.
            loader: Loader strategy identifier (``"json"``, ``"npy"``,
                ``"npz"``, ``"pt"``, etc.).
            **loader_kwargs: Extra kwargs forwarded to the loader.

        Returns:
            Any: The loaded artifact value.

        Raises:
            MissingContextError: If ``self.past_context`` is not yet
                loaded, or *step* is missing / lacks ``run_id``.
            ValueError: If *loader* is not recognised by
                :meth:`BasePlugin._load_artifact`.
        """
        if not hasattr(self, "past_context"):
            raise MissingContextError(
                "past_context is not loaded; call run() with past_run_id first"
            )
        step_entry = self.past_context.get(step)
        if not isinstance(step_entry, dict) or "run_id" not in step_entry:
            raise MissingContextError(f"Past context is missing step '{step}' or its run_id")
        spec = ArtifactSpec(
            step=step,
            filename=filename,
            attr="",
            loader=loader,
            loader_kwargs=loader_kwargs,
        )
        mlflow_loader = MLflowRunLoader(step_entry["run_id"])
        return self._load_artifact(mlflow_loader, spec)
