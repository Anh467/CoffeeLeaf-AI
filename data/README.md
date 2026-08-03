# Dataset location

Place the dataset at `data/raw` with this structure:

```text
data/raw/
├── Train/
│   ├── Healthy/
│   ├── Miner/
│   ├── Phoma/
│   └── Rust/
└── test/
    ├── Healthy/
    ├── Miner/
    ├── Phoma/
    └── Rust/
```

The images are intentionally not stored in Git. After copying them, run:

```powershell
dvc add data/raw
git add data/raw.dvc .gitignore
```

Configure a DVC remote before `dvc push`. Keep remote credentials in environment
variables or `.dvc/config.local`, never in the committed `.dvc/config` file.
