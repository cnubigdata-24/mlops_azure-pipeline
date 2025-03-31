# pip install azure-ai-ml

# In[25]:

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# 인증을 위한 Azure 클라이언트 설정
credential = DefaultAzureCredential()
client = MLClient(credential, subscription_id="<your-subscription-id>", resource_group="<your-resource-group>", workspace_name="<your-workspace-name>")

# Workspace 연결 확인
workspace = client.workspaces.get()
print('Workspace name: ' + workspace.name, 
      'Azure region: ' + workspace.location, 
      'Subscription id: ' + workspace.subscription_id, 
      'Resource group: ' + workspace.resource_group, sep='\n')

# In[26]:

from azure.ai.ml.entities import Experiment

# 실험 등록
experiment = client.experiments.create_or_update(name="diabetes-experiment")

# In[27]:

# 데이터셋 준비
from azure.ai.ml.data import Input

# 데이터셋 준비 및 분할
from sklearn.model_selection import train_test_split
from azureml.opendatasets import Diabetes

x_df = Diabetes.get_tabular_dataset().to_pandas_dataframe().dropna()
y_df = x_df.pop("Y")

X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=66)

print(X_train)

# In[28]:
# 모델 학습 및 로깅, 모델 파일 업로드
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import joblib
import math

alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for alpha in alphas:
    run = experiment.start_logging()
    run.log("alpha_value", alpha)

    model = Ridge(alpha=alpha)
    model.fit(X=X_train, y=y_train)

    y_pred = model.predict(X=X_test)
    
    rmse = math.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred))
    run.log("rmse", rmse)

    model_name = "model_alpha_" + str(alpha) + ".pkl"
    filename = "outputs/" + model_name

    joblib.dump(value=model, filename=filename)
    run.upload_file(name=model_name, path_or_stream=filename)
    run.complete()

    print(f"{alpha} exp completed")

# In[29]:

experiment

# In[30]:

# Best model 탐색 후 다운로드
minimum_rmse_runid = None
minimum_rmse = None

for run in experiment.get_runs():
    run_metrics = run.get_metrics()
    run_details = run.get_details()
    run_rmse = run_metrics["rmse"]
    run_id = run_details["runId"]
    
    if minimum_rmse is None:
        minimum_rmse = run_rmse
        minimum_rmse_runid = run_id
    else:
        if run_rmse < minimum_rmse:
            minimum_rmse = run_rmse
            minimum_rmse_runid = run_id

print("Best run_id: " + minimum_rmse_runid)
print("Best run_id rmse: " + str(minimum_rmse))
from azure.ai.ml.entities import Job
best_run = client.jobs.get(minimum_rmse_runid)
print(best_run.get_file_names())
best_run.download_file(name=str(best_run.get_file_names()[0]))

# In[31]:

# DataStore 에 Input/Output 데이터셋 등록
import numpy as np
from azure.ai.ml.entities import Data

np.savetxt('features.csv', X_train, delimiter=',')
np.savetxt('labels.csv', y_train, delimiter=',')

datastore = client.datastores.get(name='workspaceblobstore')
datastore.upload_files(files=['./features.csv', './labels.csv'],
                       target_path='diabetes-experiment/')

input_dataset = Data.Tabular.from_delimited_files(path=[(datastore, 'diabetes-experiment/features.csv')])
output_dataset = Data.Tabular.from_delimited_files(path=[(datastore, 'diabetes-experiment/labels.csv')])

# In[32]:

# Best model 등록
import sklearn

from azure.ai.ml.entities import Model
from azure.ai.ml.entities import ResourceConfiguration

model = Model.register(workspace=workspace,
                       model_name='diabetes-experiment-model',
                       model_path=f"./{str(best_run.get_file_names()[0])}", 
                       model_framework="scikit-learn",  
                       model_framework_version=sklearn.__version__,  
                       sample_input_dataset=input_dataset,
                       sample_output_dataset=output_dataset,
                       resource_configuration=ResourceConfiguration(cpu=1, memory_in_gb=0.5),
                       description='Ridge regression model to predict diabetes progression.',
                       tags={'area': 'diabetes', 'type': 'regression'})

print('Name:', model.name)
print('Version:', model.version)

# In[42]:

# 모델 배포
from azure.ai.ml.entities import AciWebservice, InferenceConfig, Environment

# Environment 설정
env = Environment.get(workspace=workspace, name="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu")

inference_config = InferenceConfig(
    entry_script="src/score.py",
    environment=env
)

# ACI 설정
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service_name = 'diabetes-service'

from azure.ai.ml.entities import Webservice
try:
    service = Webservice(workspace=workspace, name=service_name)
    service.delete()
    print("기존 서비스 삭제 완료")
except Exception as e:
    print("기존 서비스가 없거나 이미 삭제됨:", e)

# Deploy Model
service = Model.deploy(
    workspace=workspace,
    name=service_name,
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config
)

service.wait_for_deployment(show_output=True)

# In[46]:

# 배포 서비스 테스트 : 노트북
print("\n===== 모델 예측 테스트 =====")

import json

test_samples = X_test[0:2].values.tolist()
print(f"입력 데이터: {test_samples}")

input_payload = json.dumps({
    'data': test_samples
})

# 예측 실행
print("예측 요청 중...")
output = service.run(input_payload)

print("예측 결과:")
print(output)

# In[44]:

# 10. 등록된 모델 목록 확인
print("\n===== 등록된 모델 목록 =====")

registered_models = client.models.list()
for model in registered_models:
    print(f"Name: {model.name}, Version: {model.version}, Created: {model.creation_context.created_at}")

# In[ ]:

# 11. 리소스 정리 (선택 사항)
# print("\n===== 리소스 정리 =====")
# print(f"엔드포인트 {endpoint_name} 삭제 중...")
# ml_client.online_endpoints.begin_delete(name=endpoint_name)
# print("엔드포인트 삭제 완료")
