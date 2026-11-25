from pathlib import Path 
from huggingface_hub import snapshot_download


# Pinning the current version, so that future Hub updates cannot silently change the experiment
VSTAR_REVISION = "d9ae62c903da0c98336e85c5ee89cd863b04b4da"


def get_vstar_root() -> Path:
    return Path(
        snapshot_download(
            repo_id="craigwu/vstar_bench",
            repo_type="dataset",
            revision=VSTAR_REVISION,
            local_dir=Path("data/raw/vstar_bench"),
            allow_patterns=[
                "direct_attributes/*",
                "relative_position/*",
                "test_questions.jsonl",
            ],
    )
)

print(get_vstar_root())