# ADHD200

Homepage: https://fcon_1000.projects.nitrc.org/indi/adhd200/

## 1. Download source data

Download source imaging and phenotypic data from the fcp-indi S3 bucket

```bash
# all imaging data
aws s3 sync --no-sign-request s3://fcp-indi/data/Projects/ADHD200/RawDataBIDS data/RawDataBIDS

# just phenotypic
aws s3 sync --no-sign-request s3://fcp-indi/data/Projects/ADHD200/RawDataBIDS data/phenotypic/ \
  --exclude '*' \
  --include '*_phenotypic.csv'
```

## 2. Preprocessing

We preprocess the data with [fMRIPrep](https://fmriprep.org) (v25.2.3).

First get a list of all subjects

```bash
bash scripts/find_subjects.sh
```

Then preprocess each subject independently by running [`scripts/preprocess.sh`](scripts/preprocess.sh). For example

```bash
parallel -j 64 ./scripts/preprocess.sh {} ::: {1..961}
```

## 3. Curation

Construct a curated subset of complete data. The inclusion criteria are:

- runs with complete preprocessing outputs in CIFTI 91k and MNI 152 (NLin6Asym)
- runs with at least 5 min data
- subjects with at least one valid run
- subjects with complete phenotypic info
- sites with at least 20 valid subjects

We split subjects into train, validation, and test sets with a ratio of 70:15:15, stratifying by sex and diagnosis. We merge the three ADHD subdiagnosis categories into a single ADHD group. We use only the first fMRI run per subject.

| split      |   Total |   ADHD |   Control |   Female |   Male |
|:-----------|--------:|-------:|----------:|---------:|-------:|
| train      |     301 |    131 |       170 |      128 |    173 |
| validation |      64 |     28 |        36 |       27 |     37 |
| test       |      65 |     28 |        37 |       28 |     37 |


See [`scripts/adhd200_curation.ipynb`](scripts/adhd200_curation.ipynb).

```bash
uv run jupyter execute --inplace scripts/adhd200_curation.ipynb
```

The output table of curated subjects with phenotypic labels is in [`metadata/ADHD200_curated.csv`](metadata/ADHD200_curated.csv).


## 4. Generate Arrow datasets

Finally, generate output Arrow datasets for all target standard spaces

```bash
uv run python scripts/make_adhd200_arrow.py --space schaefer400
```

The script standardizes all runs to TR = 2.0s across sites, and selects the first 150 TRs (5 min) for each included run.

The script can also be run with remote data and output directories. See [`make_adhd200_arrow.sh`](scripts/make_adhd200_arrow.sh).
