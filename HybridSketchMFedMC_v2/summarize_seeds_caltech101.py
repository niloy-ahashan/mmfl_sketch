"""summarize_seeds.py for Caltech101, which it does not know the modality count of.

Only adds that count, so the "#Modalities" column reads e.g. 2/3 (67%). Called at the end of
run_caltech101.sh; takes the same flags as summarize_seeds.py.
"""
import summarize_seeds

summarize_seeds.TOTAL_MODALITIES["caltech101_mm"] = 3
summarize_seeds.TOTAL_MODALITIES["caltech101_mm_c20"] = 3

if __name__ == "__main__":
    summarize_seeds.main()
