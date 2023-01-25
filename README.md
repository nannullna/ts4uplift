# Backend-TS Dataset

Official repository of the paper: **Modeling Uplift from Observational Time-Series in Continual Scenarios** (Oral presentation, AAAI 2023 Bridge Program for Continual Causality)

[https://openreview.net/forum?id=pKyB5wMnTiy](https://openreview.net/forum?id=pKyB5wMnTiy)

## 초록 - Abstract

As the importance of causality in machine learning grows, we expect the model to learn the correct causal mechanism for robustness even under distribution shifts. Since most of the prior benchmarks focused on vision and language tasks, domain or temporal shifts in causal inference tasks have not been well explored. To this end, we introduce Backend-TS dataset for modeling uplift in continual learning scenarios. We build the dataset with CRUD data and propose continual learning tasks under temporal and domain shifts.


## 소개 - Introduction

The **Backend-TS** dataset has been developed in collaboration with [AFI Inc.](https://www.afidev.com/), a leading Backend-as-a-Service (BaaS) company specializing in mobile games, to aid research in log-level time-series and causal inference. The dataset has been named after the service provided by AFI Inc., [thebackend.io](https://www.thebackend.io/), which refers to the time-series of backend server logs. As of January 2023, this service has been utilized by 3,174 mobile games, generating a cumulative user base of over 60 million, and 115 million daily transactions. By identifying valuable patterns in this log-level data, we aim to predict user behavior in order to improve the operations of game services.

The log information of most internet services is generated on a massive scale on a daily basis, however, its utility is not commensurate with its scale. As such, this log data is commonly stored in data warehouses, with only a small proportion utilized for data science and analysis. The analysis is carried out from macro-level perspectives, such as service and company level, while individual user interests or predictions receive relatively less importance. However, the task of interest in this dataset is to predict individual user behavior, ultimately contributing to improved user satisfaction and service quality. Additionally, from the perspective of machine learning and data mining, the dataset has been created with minimal pre-processing in order to reduce dependence on manual feature engineering and selection, and to incorporate more information.


## 태스크 설명 - Task Description

The task is composed of three subtasks, ID (In-distribution), TS (Temporal Shift), OOD (Out-of-domain), and the games used for uplift modeling are A, B, and C. This is summarized in the table below.

|   Task   |           Train set           |          Valid set           |  Test set  |
|:--------:|:------------------------------|:-----------------------------|:----------:|
|    ID    | Game A APR + MAY              | Game A APR + MAY (20% split) |      -     |
|    TS    | Game A APR + MAY              | Game A APR + MAY (20% split) | Game A JUN |
|  OOD w/  | Game A APR + MAY & Game B JUN | Game B JUN (20% split)       | Game B JUL |
|  OOD w/o | Game A APR + MAY              | Game A APR + MAY (20% split) | Game C JUL |

* For ID, the performance is measured on the validation set, which is randomly sampled from the entire dataset of game A. The rest is used as the training set. Since the training and validation sets are collected from the same game and time period, we can assume that the distribution is the same, and this is used as the baseline performance measurement for the uplift model.

* For TS, the performance is evaluated on the test set, which is collected from a time point one month after the training data. This is more realistic than ID, as the uplift model predicts the behavior of future users based on a model trained on past data. If the model has learned robust features that are independent of the time axis, the performance will be similar to ID.

* For OOD, the performance is evaluated on the test set, which is collected from games B and C. This is a more general evaluation of the model's ability to predict the behavior of users from games other than the training data. OOD is divided into two subtasks.

    * For OOD w/, the new training data from game B is given, and the model is fine-tuned on the new game. The performance is evaluated on the test set, which is collected from a time point after the training data. This is a more general evaluation of the model's ability to predict the behavior of users under a different domain.

    * For OOD w/o, the performance is evaluated on the test set, which is collected from game C. This evaluates the model's ability to predict the behavior of users from unseen games, and therefore, the model is not fine-tuned on the new game. If the model has learned robust features that are independent of the domain, the performance will not drop significantly.


## 데이터셋 설명 - Dataset Description


## 한계 - Limitations




## 추가 자료 - Additional Resources




