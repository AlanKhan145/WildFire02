# WildFire01

Patch-wise GNN pipeline for next-day wildfire spread prediction using the Kaggle dataset `fantineh/next-day-wildfire-spread`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

Run each script from the `WildFire01` directory:

```bash
python scripts/00_download_data.py --config configs/base.yaml
python scripts/01_count_records.py --config configs/base.yaml --data_dir <DATA_DIR>
python scripts/02_compute_train_stats.py --config configs/base.yaml --data_dir <DATA_DIR>
python scripts/10_hpo_gnn.py --config configs/base.yaml --hpo_config configs/gnn_hpo.yaml --data_dir <DATA_DIR>
python scripts/11_train_gnn_final.py --config configs/base.yaml --final_config configs/gnn_final.yaml --data_dir <DATA_DIR>
python scripts/20_sweep_thresholds.py --config configs/base.yaml --data_dir <DATA_DIR>
python scripts/21_feature_ablation.py --config configs/base.yaml --data_dir <DATA_DIR>
python scripts/30_vis_largest_fire_tile.py --config configs/base.yaml --data_dir <DATA_DIR>
python scripts/31_vis_best_worst_tiles.py --config configs/base.yaml --data_dir <DATA_DIR>
python scripts/32_vis_multi_resolution.py --config configs/base.yaml --data_dir <DATA_DIR>
```

Or run the full pipeline:

```bash
python scripts/99_run_all.py --config configs/base.yaml --data_dir <DATA_DIR>
```

## Notes

- Normalization statistics are stored in `artifacts/data/DATA_STATS_train.json`.
- HPO outputs (`df_hpo.csv`, `best_cfg.json`) are stored in `artifacts/gnn/`.
- Final model is saved to `artifacts/gnn/model_final.keras`.
- Reports and plots are saved under `artifacts/reports/`.
