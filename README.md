# CARRA

Implementation of **CARRA: Conflict-Aware Rank-Based Replay Allocation for Data-Free Class-Incremental Skeleton-Based Gesture Recognition**.

## Installation

Requires Python >= 3.8. Package versions are listed in [requirements.txt](requirements.txt).

```bash
# Clone the repository
git clone https://github.com/QienJane/CARRA.git
cd CARRA

# Install dependencies
python -m pip install -r requirements.txt
```

## Data Preparation

Download the skeleton data below, then run the preparation commands from the project root.

### SHREC'17 Track

SHREC'17 contains 14 hand gesture classes from 28 subjects, with 22 joints per frame.

1. Download the dataset from the [official website](http://www-rech.telecom-lille.fr/shrec2017-hand/).
2. Extract it to `datasets/hgr_shrec_2017/`, keeping `train_gestures.txt`, `test_gestures.txt`, and the `gesture_*` directories in their original layout. Preparation uses the provided split and 14-class labels.

```bash
python -B run_carra.py --dataset hgr_shrec_2017 --prepare
```

### EgoGesture3D

EgoGesture3D provides 42-joint hand skeletons for 83 gesture classes from 50 subjects.

1. Download the [EgoGesture3D skeletons](https://drive.google.com/file/d/1pHE0Q9MtVS5BLaV2CBN1rLP_Ed7nvfac/view) released by [DFCIL-HGR](https://github.com/humansensinglab/dfcil-hgr#datasets). The original RGB-D data are available on the [EgoGesture website](https://nlpr.ia.ac.cn/iva/yfzhang/datasets/egogesture.html).
2. Extract the skeleton package to `datasets/ego_gesture/`, keeping the CSV files and `split_files/{train,val,test}.txt` in their original layout.
   Preparation uses the official subject-disjoint splits.

```bash
python -B run_carra.py --dataset ego_gesture --prepare
```

### NTU RGB+D 60

NTU RGB+D 60 contains 60 action classes with 25 joints per skeleton. We use the cross-subject split.

1. Download the skeleton data from the [official website](https://rose1.ntu.edu.sg/dataset/actionRecognition/).
2. Extract the raw `.skeleton` files to `datasets/ntu60/`, keeping their original filenames.

```bash
python -B run_carra.py --dataset ntu60 --prepare
```

### NTU RGB+D 120

NTU RGB+D 120 extends NTU-60 to 120 action classes. We use the cross-subject split.

1. Download both skeleton packages, **S001–S017** and **S018–S032**, from the [official website](https://rose1.ntu.edu.sg/dataset/actionRecognition/).
2. Extract both packages to `datasets/ntu120/`, keeping the original `.skeleton` filenames.

```bash
python -B run_carra.py --dataset ntu120 --prepare
```

Both NTU datasets use the bundled missing-skeleton lists to exclude invalid samples.

### NW-UCLA

NW-UCLA contains 10 action classes with 20 joints per skeleton. Views 1 and 2 are used for training and view 3 for testing.

1. The original dataset is available on the [official website](https://wangjiangb.github.io/my_data.html). For this implementation, download the [preprocessed JSON files](https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?dl=0) linked by [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN#nw-ucla).
2. Extract the package so the JSON files are under `datasets/nw_ucla/all_sqe/`.

```bash
python -B run_carra.py --dataset nw_ucla --prepare
```

Prepared data are saved in each dataset's `prepared/` directory. For a custom location, pass the same `--data-root /path/to/data` to preparation and training.

## Training

A single run trains the base task followed by all incremental tasks:

```bash
python -B run_carra.py --dataset hgr_shrec_2017 --gpu 0
```

The default protocol is `main`, with both seeds set to `0`. Settings and task schedules are in [configs/](configs/); use `--config /path/to/overrides.yaml` to override settings. For NTU, `--protocol` also accepts `c5`, `c10`, or `c20`.

Run all 30 combinations of 10 class orders and 3 initialization seeds sequentially:

```bash
python -B run_carra.py --dataset hgr_shrec_2017 --gpu 0 \
  --all-seeds --output-root outputs/all_seeds
```

Use `--gpu cpu` to run on CPU. To repeat the same run, choose a new `--output-root`; nonempty run directories are not overwritten.

## Evaluation

Evaluation runs automatically after each task on all seen classes. There is no separate testing or resume command.

Results are saved in `outputs/<dataset>/<protocol>/order_<seed>_init_<seed>/`, including per-task metrics, replay plans, the latest `state.pt`, and `summary.json`. After `--all-seeds` completes, `<output-root>/<dataset>/<protocol>/aggregate_30_runs.json` contains means and sample standard deviations.

## Project Structure

```text
CARRA/
├── configs/          # Settings and task schedules
├── datasets/         # Data loading and preparation
├── graph/            # Skeleton topologies
├── losses/           # Supervised contrastive loss
├── model_defs/       # Backbone
├── utils/            # Replay, memory, training, and metrics
├── run_carra.py      # Main entry point
└── requirements.txt
```

## Acknowledgements

CARRA builds on [BOAT-MI](https://github.com/humansensinglab/dfcil-hgr) and [TD-GCN](https://github.com/liujf69/TD-GCN-Gesture). We thank the authors for their work.

## Contact

For questions, please contact zhanqq@mail2.sysu.edu.cn.
