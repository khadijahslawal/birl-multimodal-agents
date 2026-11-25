birl-multimodal-agents/
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── .env.example
│
├── configs/
│   ├── data/
│   │   └── vstar.yaml
│   └── experiments/
│       └── pilot.yaml
│
├── data/
│   ├── README.md
│   ├── raw/                 # Downloaded snapshot; gitignored
│   │   └── vstar_bench/
│   ├── processed/           # Generated evidence levels; gitignored
│   └── manifests/           # Small versioned metadata and splits
│
├── scripts/
│   ├── download_vstar.py
│   ├── validate_vstar.py
│   ├── preview_evidence_levels.py
│   ├── run_forced_level_baselines.py
│   └── collect_trajectories.py
│
├── src/
│   └── birl_multimodal_agents/
│       ├── data/
│       │   ├── schema.py
│       │   └── vstar.py
│       ├── envs/
│       │   ├── crops.py
│       │   └── progressive_evidence.py
│       ├── models/
│       │   ├── base.py
│       │   └── hf_vlm.py
│       ├── trajectories/
│       │   ├── schema.py
│       │   └── collection.py
│       ├── irl/
│       │   ├── features.py
│       │   ├── maxent.py
│       │   └── birl.py
│       └── evaluation/
│           ├── calibration.py
│           ├── metrics.py
│           └── bootstrap.py
│
├── tests/
│   ├── data/
│   │   └── test_vstar.py
│   ├── envs/
│   │   └── test_progressive_evidence.py
│   └── irl/
│       └── test_reward_recovery.py
│
├── notebooks/               # Exploration only
└── outputs/                 # Logs, figures, posteriors; gitignored