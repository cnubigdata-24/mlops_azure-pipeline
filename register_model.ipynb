from azureml.core import Workspace
from azureml.core.model import Model

ws = Workspace(subscription_id="<your subscription_id>",
               resource_group="<your resource_group_name>",
               workspace_name="mlops-pipeline")

model = Model.register(
    ws, 
    model_name="model", 
    model_path="./artifacts/model.joblib",
    tags={'area': "wine", 'type': "regression"},
    description="RandomForest regression model to predict wine quality"
    )

scaler = Model.register(
    ws, 
    model_name=f"scaler", 
    model_path=f"./artifacts/scaler.joblib")

print(f"wine_model - name: {model.name}, id: {model.id}, ver: {model.version}")
print(f"wine_scaler - name: {scaler.name}, id: {scaler.id}, ver: {scaler.version}")
