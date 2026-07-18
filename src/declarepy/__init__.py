"""declarepy: a transparent Python translation of the DeclareDesign engine.

Declare a design (Model → Inquiry → Data strategy → Answer strategy),
diagnose it by Monte-Carlo simulation, and redesign it over parameter
sweeps — with explicit callables, explicit potential-outcome columns, and
reproducible seeded generators throughout.

Attribution: an independent translation of concepts from Blair, Coppock &
Humphreys (2023), *Research Design in the Social Sciences*, and the
DeclareDesign R ecosystem (declaredesign.org). Not affiliated with or
endorsed by the DeclareDesign authors. MIT licensed; see LICENSE.
"""

from . import course, data
from ._rng import resolve_rng
from .diagnose import (
    DesignGrid,
    Diagnosands,
    Diagnosis,
    RunResult,
    diagnose,
    diagnose_all,
    draw_data,
    redesign,
    run_design,
)
from .estimators import (
    EstimatorResult,
    difference_in_means,
    glm_logit,
    lm_robust,
    logit_ame,
    prop_test,
)
from .ra import block_ra, block_rs, complete_ra, complete_rs, simple_ra, simple_rs
from .steps import (
    Assignment,
    Design,
    DesignStep,
    Estimator,
    Inquiry,
    Measurement,
    Model,
    Sampling,
    potential_outcomes,
    reveal_outcomes,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # steps
    "Design", "DesignStep", "Model", "Inquiry", "Sampling", "Assignment",
    "Measurement", "Estimator", "potential_outcomes", "reveal_outcomes",
    # engine
    "run_design", "draw_data", "diagnose", "diagnose_all", "redesign",
    "Diagnosands", "Diagnosis", "RunResult", "DesignGrid",
    # procedures
    "complete_ra", "block_ra", "simple_ra", "complete_rs", "simple_rs", "block_rs",
    # estimators
    "difference_in_means", "lm_robust", "prop_test", "glm_logit", "logit_ame",
    "EstimatorResult",
    # utilities
    "course", "data", "resolve_rng",
]
