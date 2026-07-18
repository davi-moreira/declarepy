"""The declaration library: Python translations of the book's designs.

Each ``declaration_X_Y`` factory returns a fresh :class:`declarepy.Design`
translating the corresponding ``declaration_X.Y.R`` from the replication
archive of *Research Design in the Social Sciences* (Blair, Coppock &
Humphreys 2023). Factories take the same parameters the R sources sweep in
``redesign()`` calls, so ``dp.redesign(declaration_11_1, N=[100, 200])``
mirrors the book's usage.

Every translation carries a provenance line and is validated against the
book's saved diagnosis objects per docs/spec/VALIDATION_REPORT.md; per-check
results live in ``validation/t3_results_*.csv``.
"""

from .ch02 import declaration_2_1, declaration_2_2
from .ch04 import declaration_4_1
from .ch05 import declaration_5_1
from .ch07 import declaration_7_1
from .ch09 import (
    declaration_9_1,
    declaration_9_2,
    declaration_9_3,
    declaration_9_4,
    declaration_9_5,
    declaration_9_6,
    declaration_9_7,
)
from .ch10 import (
    declaration_10_1,
    declaration_10_2,
    declaration_10_3a,
    declaration_10_3b,
    declaration_10_4,
    declaration_10a,
)
from .ch11 import (
    declaration_11_1,
    declaration_11_2,
    declaration_11_3,
    declaration_11_4,
    declaration_11_5,
)
from .ch12 import declaration_12_1
from .ch13 import declaration_13_1, declaration_13_2
from .ch15 import (
    declaration_15_1,
    declaration_15_2,
    declaration_15_3,
    declaration_15_4,
    declaration_15_5,
    declaration_15_6,
)
from .ch16 import (
    declaration_16_1,
    declaration_16_2,
    declaration_16_3,
    declaration_16_4,
    declaration_16_5,
    declaration_16_6,
)
from .ch17 import (
    declaration_17_1,
    declaration_17_2,
    declaration_17_3,
    declaration_17_4,
    declaration_17_5,
    declaration_17_6,
)
from .ch18 import (
    declaration_18_1,
    declaration_18_2,
    declaration_18_3,
    declaration_18_4,
    declaration_18_5,
    declaration_18_6,
    declaration_18_7,
    declaration_18_8,
    declaration_18_9_encouragement,
    declaration_18_9_placebo,
    declaration_18_9a,
    declaration_18_10,
    declaration_18_11,
    declaration_18_12,
    declaration_18_13,
)
from .ch19 import (
    declaration_19_1,
    declaration_19_2,
    declaration_19_3,
    declaration_19_4,
)
from .ch21 import declaration_21a, declaration_21b
from .ch23 import declaration_23_1

__all__ = [
    "declaration_2_1", "declaration_2_2",
    "declaration_4_1", "declaration_5_1", "declaration_7_1",
    "declaration_9_1", "declaration_9_2", "declaration_9_3", "declaration_9_4",
    "declaration_9_5", "declaration_9_6", "declaration_9_7",
    "declaration_10_1", "declaration_10_2", "declaration_10_3a",
    "declaration_10_3b", "declaration_10_4", "declaration_10a",
    "declaration_11_1", "declaration_11_2", "declaration_11_3",
    "declaration_11_4", "declaration_11_5",
    "declaration_12_1", "declaration_13_1", "declaration_13_2",
    "declaration_15_1", "declaration_15_2", "declaration_15_3",
    "declaration_15_4", "declaration_15_5", "declaration_15_6",
    "declaration_16_1", "declaration_16_2", "declaration_16_3",
    "declaration_16_4", "declaration_16_5", "declaration_16_6",
    "declaration_17_1", "declaration_17_2", "declaration_17_3",
    "declaration_17_4", "declaration_17_5", "declaration_17_6",
    "declaration_18_1", "declaration_18_2", "declaration_18_3",
    "declaration_18_4", "declaration_18_5", "declaration_18_6",
    "declaration_18_7", "declaration_18_8", "declaration_18_9a",
    "declaration_18_9_encouragement", "declaration_18_9_placebo",
    "declaration_18_10", "declaration_18_11", "declaration_18_12",
    "declaration_18_13",
    "declaration_19_1", "declaration_19_2", "declaration_19_3",
    "declaration_19_4",
    "declaration_21a", "declaration_21b",
    "declaration_23_1",
]
