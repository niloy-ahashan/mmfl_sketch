"""summarize_seeds.py for PTB-XL, which it does not know the modality count of.

Only adds that count, so the "#Modalities" column reads e.g. 6/12 (50%). Called at the end of
run_ptbxl_12mod.sh; takes the same flags as summarize_seeds.py.
"""
import summarize_seeds

summarize_seeds.TOTAL_MODALITIES["ptbxl_12mod_mm"] = 12

if __name__ == "__main__":
    summarize_seeds.main()
