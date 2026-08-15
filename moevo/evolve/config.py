"""SkyDiscover configuration builder."""

from __future__ import annotations

try:
    from skydiscover.config import (
        _DB_CONFIG_BY_TYPE,
        AdaEvolveDatabaseConfig,
        Config,
        ContextBuilderConfig,
        EvaluatorConfig,
        LLMConfig,
        LLMModelConfig,
        MonitorConfig,
        SearchConfig,
    )
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "The SkyDiscover scalar-selection baseline requires the separate "
        "`skydiscover` package, which is not distributed with moevo:\n"
        "    https://github.com/skydiscover-ai/skydiscover\n"
        "It is needed only to reproduce the SkyDiscover rows of Tables 1-3. "
        "The MOEvo engine itself has no such dependency -- run with "
        "`--engine moevo` (the default)."
    ) from exc

# The mutation system prompt is defined once, in the engine, and reused here so
# the two engines cannot silently drift apart -- it is the experimental control
# in the Pareto-vs-scalar comparison.
from moevo.generation.prompt import SYSTEM_MESSAGE  # noqa: E402

__all__ = ["SYSTEM_MESSAGE", "build_config"]


def build_config(model: str, iterations: int, search: str = "adaevolve") -> Config:
    """Build a SkyDiscover Config for the chosen search strategy."""
    db_config = _DB_CONFIG_BY_TYPE.get(search, AdaEvolveDatabaseConfig)()

    if search == "adaevolve":
        db_config = AdaEvolveDatabaseConfig(
            population_size=10,
            num_islands=2,
            decay=0.9,
            intensity_min=0.2,
            intensity_max=0.6,
            use_adaptive_search=True,
            use_ucb_selection=True,
            use_migration=True,
            use_unified_archive=True,
            migration_interval=5,
            migration_count=2,
            use_dynamic_islands=False,
            use_paradigm_breakthrough=False,
            enable_error_retry=True,
            max_error_retries=1,
            archive_size=20,
            stagnation_threshold=5,
            stagnation_multi_child_count=2,
        )

    return Config(
        max_iterations=iterations,
        checkpoint_interval=max(1, iterations // 2),
        log_level="INFO",
        language="python",
        file_suffix=".py",
        diff_based_generation=True,
        max_solution_length=80000,
        max_parallel_iterations=1,
        llm=LLMConfig(
            models=[LLMModelConfig(name=model)],
            temperature=0.7,
            max_tokens=32000,
            timeout=180,
            retries=2,
            retry_delay=5,
        ),
        context_builder=ContextBuilderConfig(system_message=SYSTEM_MESSAGE),
        search=SearchConfig(type=search, num_context_programs=2, database=db_config),
        evaluator=EvaluatorConfig(timeout=1200, max_retries=1, cascade_evaluation=False),
        monitor=MonitorConfig(enabled=False),
    )
