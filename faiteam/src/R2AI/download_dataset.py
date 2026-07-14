"""Download Vietnamese Legal Documents dataset from Hugging Face."""

from pathlib import Path

from datasets import load_dataset

DATASET_ID = "vohuutridung/vietnamese-legal-documents"
OUTPUT_DIR = Path(__file__).parent / "data" / "vietnamese-legal-documents"


def download_config(config_name: str) -> None:
    out = OUTPUT_DIR / config_name
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Downloading config: {config_name} -> {out} ===")
    ds = load_dataset(DATASET_ID, config_name, trust_remote_code=True)
    ds["data"].to_parquet(str(out / "data.parquet"))
    print(f"Saved {len(ds['data']):,} rows to {out / 'data.parquet'}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for config in ("metadata", "content"):
        download_config(config)
    print("\nDone.")


if __name__ == "__main__":
    main()
