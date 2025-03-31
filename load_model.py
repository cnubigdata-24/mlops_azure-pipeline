from azureml.core import Workspace
from azureml.core.model import Model

ws = Workspace(subscription_id="<your subscription_id>",
               resource_group="<your resource_group_name>",
               workspace_name="mlops-pipeline")
wine_model = Model(
    ws, 
    'model', 
    version=1)

wine_scaler = Model(
    ws, 
    'scaler', 
    version=1
)

print(wine_model)
