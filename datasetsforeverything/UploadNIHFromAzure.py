from azureml.core import Workspace, Dataset

subscription_id = '2ef81ad0-e0cd-457d-a806-8f0213cdbe4e'
resource_group = 'Thevindu_ML'
workspace_name = 'MSOE_ML_2'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='NIH_COMPLETE')
dataset.download(target_path='C:/Users/samee/Downloads', overwrite=False)