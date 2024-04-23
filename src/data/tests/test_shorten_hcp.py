from datetime import datetime

from src.data.load_data import create_shortened_hcp_data

data_dict = {
  "train": [
    "/rest1/sub-102816/ses-01/sub-102816_task-rest1_ses-01_desc-timeseries_scale-3",
    "/rest1/sub-107422/ses-01/sub-107422_task-rest1_ses-01_desc-timeseries_scale-3",
    "/rest1/sub-113316/ses-01/sub-113316_task-rest1_ses-01_desc-timeseries_scale-3",
    "/rest1/sub-116524/ses-01/sub-116524_task-rest1_ses-01_desc-timeseries_scale-3",
    "/rest1/sub-118528/ses-01/sub-118528_task-rest1_ses-01_desc-timeseries_scale-3",
    "/rest1/sub-119833/ses-01/sub-119833_task-rest1_ses-01_desc-timeseries_scale-3",
    "/rest1/sub-120515/ses-01/sub-120515_task-rest1_ses-01_desc-timeseries_scale-3",
    "/rest1/sub-123117/ses-01/sub-123117_task-rest1_ses-01_desc-timeseries_scale-3",
    "/rest1/sub-124422/ses-01/sub-124422_task-rest1_ses-01_desc-timeseries_scale-3"
  ],
  "val": [
    "/rest1/sub-100408/ses-01/sub-100408_task-rest1_ses-01_desc-timeseries_scale-3",
    "/rest1/sub-135932/ses-01/sub-135932_task-rest1_ses-01_desc-timeseries_scale-3",
    "/rest1/sub-136833/ses-01/sub-136833_task-rest1_ses-01_desc-timeseries_scale-3"
  ],
  "test": [
    "/rest1/sub-104012/ses-01/sub-104012_task-rest1_ses-01_desc-timeseries_scale-3",
    "/rest1/sub-104416/ses-01/sub-104416_task-rest1_ses-01_desc-timeseries_scale-3",
    "/rest1/sub-123723/ses-01/sub-123723_task-rest1_ses-01_desc-timeseries_scale-3"
  ]
}

now = datetime.now()

create_shortened_hcp_data(data_dict,
                          "inputs/connectomes/hcp.h5",
                          f"/scratch/mschoett/test_{now.strftime('%H-%M-%S')}",
                          fraction=0.5)