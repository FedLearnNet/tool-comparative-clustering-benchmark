# tool-comparative-clustering-benchmark

Compares clustering outputs using internal quality scores and optional ground-truth labels.

## Runtime inputs

Supply input files according to the schemas in `app.yml`.

## Inputs and outputs

### Inputs

| Name | Type | Description |
| --- | --- | --- |
| `features` | CSV | Feature table as CSV, including the sample identifier column. |
| `predictions` | CSV | Prediction table as CSV, containing the sample identifier and one or more clustering result columns. |
| `ground_truth` | CSV | Optional ground truth table as CSV, containing the sample identifier and true label column. |

### Outputs

| Name | Type | Description |
| --- | --- | --- |
| `output` | CSV | Benchmark scores for all clustering prediction columns as CSV. |

## Configuration

| Name | Default | Description |
| --- | --- | --- |
| `id_column` | `sample_id` | Column name used as unique sample identifier. |
| `truth_label_column` | `cluster_label` | Default label column name for clustering predictions. |
| `sort_by` | `silhouette` | Metric used to rank clustering models. |

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
# Set APP_ID and platform URLs before connecting.
python main.py
```

The Docker base image provides the FL-Net API. `requirements.txt` lists additional
runtime dependencies; `requirements-dev.txt` includes the API for local development.
Runtime datasets and trained models must be supplied separately; see `data/README.md`.

Build with `docker build -t tool-comparative-clustering-benchmark .`.
