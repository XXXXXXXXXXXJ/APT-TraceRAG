import json
from pathlib import Path

FINAL_DIR = Path("./Executable/R_Index_final")
BEHAVIOUR_DIR = Path("./Executable/R_behaviour")
OUTPUT_DIR = Path("./Executable/R_all")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def remove_scans(data: dict) -> dict:
    """
    Remove the scans field (if present).
    """
    cleaned = data.copy()
    cleaned.pop("scans", None)
    return cleaned


def merge_json(final_data: dict, behaviour_data: dict | None) -> dict:
    """
    final_data: scans already removed
    behaviour_data: may be None
    """
    merged = final_data.copy()

    if isinstance(behaviour_data, dict):
        for k, v in behaviour_data.items():
            merged[k] = v

    return merged


def process():
    for final_file in FINAL_DIR.glob("*.json"):
        try:
            final_data = load_json(final_file)
            final_data = remove_scans(final_data)

            behaviour_file = BEHAVIOUR_DIR / final_file.name
            if behaviour_file.exists():
                behaviour_data = load_json(behaviour_file)
            else:
                behaviour_data = None

            merged = merge_json(final_data, behaviour_data)

            output_path = OUTPUT_DIR / final_file.name
            with open(output_path, "w", encoding="utf-8") as out:
                json.dump(merged, out, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"[!] Failed to process {final_file.name}: {e}")


if __name__ == "__main__":
    process()
