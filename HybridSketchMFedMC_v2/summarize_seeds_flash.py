"""summarize_seeds.py for FLASH, which it does not know the modality count of.

Only adds that count, so the "#Modalities" column reads e.g. 1/3 (33%). Called at the end of
run_flash.sh; takes the same flags as summarize_seeds.py.
"""
import summarize_seeds

summarize_seeds.TOTAL_MODALITIES["flash_mm"] = 3
summarize_seeds.TOTAL_MODALITIES["flash_mm_episode"] = 3

if __name__ == "__main__":
    summarize_seeds.main()
