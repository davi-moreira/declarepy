"""The declaration library: Python translations of the book's designs.

Each ``declaration_X_Y`` factory returns a fresh :class:`declarepy.Design`
translating the corresponding ``declaration_X.Y.R`` from the replication
archive of *Research Design in the Social Sciences* (Blair, Coppock &
Humphreys 2023). Factories take the same parameters the R sources sweep in
``redesign()`` calls, so ``dp.redesign(declaration_11_1, N=[100, 200])``
mirrors the book's usage.

Every translation carries a provenance line and is validated against the
book's saved diagnosis objects per docs/spec/VALIDATION_REPORT.md.
"""

from .ch02 import declaration_2_1, declaration_2_2
from .ch09 import declaration_9_1
from .ch10 import declaration_10_1
from .ch11 import declaration_11_1
from .ch18 import declaration_18_1

__all__ = [
    "declaration_2_1",
    "declaration_2_2",
    "declaration_9_1",
    "declaration_10_1",
    "declaration_11_1",
    "declaration_18_1",
]
