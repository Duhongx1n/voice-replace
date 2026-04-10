# -*- coding: utf-8 -*-
"""允许通过 python -m voice_replace 运行。"""

from __future__ import annotations

import sys

from voice_replace.cli import main

if __name__ == "__main__":
    sys.exit(main())
