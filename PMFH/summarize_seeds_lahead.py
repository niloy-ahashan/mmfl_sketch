"""``summarize_seeds.py`` for ``main_pmfh_lahead.py`` (PMFH + logit-adjusted fed. head).

Same columns and output as ``summarize_seeds.py``; only adds the ``PMFHla`` prefix, e.g.
  venv/bin/python PMFH/summarize_seeds_lahead.py --prefix PMFHla \
    --dataset ptbxl_mm_p44 --seeds 42 43 44 45 46
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import summarize_seeds

summarize_seeds.METHODS.update({
    "PMFHla": "PMFH (fed. head, logit-adj.)",
})

if __name__ == "__main__":
    summarize_seeds.main()
