# Will this hotel booking be cancelled?

Final project for **FEE308 Data Mining** (İstanbul Topkapı University, Fall 2026–27).

We predict, at the moment a booking is made, whether it will be cancelled. A hotel's revenue
manager could use the predicted risk to decide where to overbook, when to ask for a deposit
and which guests to contact before arrival.

| | |
|---|---|
| **Route** | Classification (binary) |
| **Target** | `is_canceled`: 1 = cancelled or no-show (37.0%), 0 = stayed (63.0%) |
| **Dataset** | Hotel Booking Demand [1]: 119,390 bookings × 32 columns, two hotels in Portugal, 2015–2017 |
| **Models** | Majority-class baseline, logistic regression, decision tree, k-NN |
| **Main metrics** | PR-AUC and F1 for the cancelled class (mean ± std over 5 CV folds, then one held-out test run) |

## Team

| Name | Student number | Responsibility |
|---|---|---|
| Mikhail Savushkin | 23040102083 | EDA, preprocessing pipeline, logistic regression, k-NN |
| Ahmet Sait Almış | 23040102057 | Baseline, decision tree, threshold choice and test evaluation, subgroup and leakage checks |

Group name: **NonRefundable**

## Repository structure

```
├── data/                 # raw CSV goes here (not committed, see "Getting the data")
├── notebooks/            # one notebook per member, numbered (see naming below)
├── src/
│   └── download_data.py  # downloads and checks the dataset
├── report/               # proposal and final report (PDF)
├── presentation/         # presentation video (.mp4, under 100 MB)
├── requirements.txt
└── README.md
```

## Getting the data

The dataset is not stored in this repository. Download it with:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/download_data.py
```

This saves `data/hotels.csv` and checks that it has 119,390 rows and 32 columns.

Source: the combined CSV of both hotels from the TidyTuesday repository [2], originally
published by Antonio, de Almeida and Nunes [1] under CC BY 4.0. Hotel and customer
identifiers were removed by the original authors.

## Reproducing the results

1. Get the data (above).
2. Run the notebooks in `notebooks/` in numeric order.
3. Every notebook uses the same split: stratified 80/20 with `random_state=42`. The test set is
   used only once, in the final evaluation.

## Leak-free rules we follow

- Columns known only after the outcome are dropped: `reservation_status`,
  `reservation_status_date` and `assigned_room_type`.
- Imputation, encoding and scaling live inside a scikit-learn `Pipeline`, fitted on training
  folds only.
- Hyperparameters and the decision threshold are chosen with cross-validation on the training
  set, never on the test set.

## File naming (from the Project Brief)

Files start with the name and student number of the member who submits them, then the group name:

| Deliverable | File name |
|---|---|
| Proposal | `<StudentName>_<StudentNumber>_<GroupName>_FEE308_ProjectOutline.pdf` |
| Report | `<StudentName>_<StudentNumber>_<GroupName>_FEE308_ProjectReport.pdf` |
| Notebooks | `<StudentName>_<StudentNumber>_<GroupName>_FEE308_1.ipynb`, `…_2.ipynb`, … |
| Video | `<StudentName>_<StudentNumber>_<GroupName>_FEE308_Presentation.mp4` |

Presentation video: _link TBD_

## References

[1] N. Antonio, A. de Almeida, and L. Nunes, "Hotel booking demand datasets," *Data in Brief*,
vol. 22, pp. 41–49, 2019, doi: [10.1016/j.dib.2018.11.126](https://doi.org/10.1016/j.dib.2018.11.126).

[2] R for Data Science Online Learning Community, "Hotels," TidyTuesday, GitHub, 2020. [Online].
Available: https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-02-11
