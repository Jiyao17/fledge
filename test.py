

from source.tasks.sc import SCAggregatorTask, SCTaskHelper, SCTrainerTask, SCDatasetPartitionerByUser


trainset, testset = SCTaskHelper.get_datasets("./dataset/raw/")
p = SCDatasetPartitionerByUser(trainset, None, None, None)

subsets = p.get_pfl_subsets(100, 0.3)

SCDatasetPartitionerByUser.save_subsets(subsets[:5], "./dataset/partitioned/sc/")
