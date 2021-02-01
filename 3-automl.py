#!/usr/bin/env python
# coding: utf-8

# # Automated ML

# In[1]:


import datetime
import json
import pathlib
import pprint
import requests

import joblib
from sklearn.metrics import classification_report


# In[2]:


import azureml.core
from azureml.core import Workspace, Experiment
from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.dataset import Dataset
from azureml.widgets import RunDetails


# In[3]:


from nd00333.model.automl.train import run_config as automl_train_config
from nd00333.model.deploy import run_config as deploy_config
from nd00333 import utils as package_utils


# In[4]:


print("azureml.core.VERSION", azureml.core.VERSION)


# ## Create an experiment
# 
# Create an experiment identified by the creation date. The purpose of identifying the experiments is to not mix the manually run experiments (using this jupyter notebook) with the experiments run using the deployment automation of the master git branch.
# 

# In[5]:


#!az logout


# In[5]:


workspace = Workspace.from_config()
utcnow = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H-%m")
utcnow = "2021-01-29-09-02"
experiment_name = f"jupyter-{utcnow}"


# In[6]:


experiment = Experiment(workspace, experiment_name)
print(f"experiment_name {experiment_name}")


# ## Dataset
# 
# Load the dataset from local files on disk into the default AzureML workspace datastore.
# Store the versioned (by the "_1" suffix), combined train+validate dataset for the purpose of cross-validation, and test sets under separate datastore paths. The uploading of large datasets tends to break with "The write operation timed out" when using the Python SDK and therefore the `az` is used. See https://docs.microsoft.com/en-us/answers/questions/43980/cannot-upload-local-files-to-azureml-datastore-pyt.html

# In[7]:


dataset_trainandvalidate_name = "ids2018trainandvalidate_1"
dataset_test_name = "ids2018test_1"
dataset_2017_name = "ids2017full_1"


# In[8]:


get_ipython().system('az login')


# In[9]:


get_ipython().system('az extension add --name azure-cli-ml --version 1.21.0')


# In[11]:


get_ipython().system('az --version')


# In[10]:


get_ipython().run_cell_magic('time', '', '!az ml datastore upload --name workspaceblobstore --verbose \\\n    --src-path datasets/ids2018trainandvalidate --target-path \\\n    $dataset_trainandvalidate_name')


# In[11]:


get_ipython().run_cell_magic('time', '', '!az ml datastore upload --name workspaceblobstore --verbose \\\n    --src-path datasets/ids2018test --target-path \\\n    $dataset_test_name')


# In[12]:


get_ipython().run_cell_magic('time', '', '!az ml datastore upload --name workspaceblobstore --verbose \\\n    --src-path datasets/ids2017full --target-path \\\n    $dataset_2017_name')


# Upload and register the uploaded datasets into AzureML workspace. In principle the above `az` commands should not be necessary.

# In[13]:


get_ipython().run_cell_magic('time', '', '!PYTHONPATH=. python nd00333/dataset/register/register.py \\\n    --dataset-path datasets --dataset-name ids2018trainandvalidate --dataset-version 1 \\\n    --dataset-type tabular')


# In[14]:


get_ipython().run_cell_magic('time', '', '!PYTHONPATH=. python nd00333/dataset/register/register.py \\\n     --dataset-path datasets --dataset-name ids2018test --dataset-version 1 \\\n     --dataset-type tabular')


# In[15]:


get_ipython().run_cell_magic('time', '', '!PYTHONPATH=. python nd00333/dataset/register/register.py \\\n     --dataset-path datasets --dataset-name ids2017full --dataset-version 1 \\\n     --dataset-type tabular')


# ## AutoML Configuration
# 
# It should be noted that several settings that may affect the model performance are used in AutoMLConfig in order to speed up the training:
# - the number of `iterations` (individual model runs) is limited to 15,
# - instead of a cross-validation method, a single split into the training and validation data is specified using `validation_size`=0.3 in order to reduce the training time,
# - the slow `enable_stack_ensemble` ensemble model is excluded,
# - only the "LightGBM", "LogisticRegression", "SGD", "XGBoostClassifier" models are allowed in the runs in `allowed_models`. Models from the "RandomForest" family unbounded by `max_depth` may grow very large (to e.g. several hundreds Mbytes).
# 
# The individual model runs are performed multi-threaded using all cores available on the compute instance by `max_cores_per_iteration=`-1. The number of `max_concurrent_iterations` is set to the number of the nodes in the compute cluster.
# 
# See [nd00333/model/automl/train/run_config.py](nd00333/model/automl/train/run_config.py) for more details.
# 
# The `norm_macro_recall` is used as the performance metrics. See [README.md](README.md) for the rationale.

# Perform the configuration of the AutoML run, including setting up the remote AML compute cluster

# In[16]:


get_ipython().run_cell_magic('time', '', 'automl_run_config = automl_train_config.main(\n                    dataset_trainandvalidate_name=dataset_trainandvalidate_name)')


# Submit the AutoML configuration to the compute cluster

# In[17]:


get_ipython().run_cell_magic('time', '', 'automl_run = experiment.submit(config=automl_run_config, show_output=False)')


# In[18]:


get_ipython().run_cell_magic('time', '', 'automl_run.wait_for_completion(show_output=True)')


# In[20]:


assert(automl_run.get_status() == "Completed")


# ## Run Details

# The widget below shows the first few best runs of the experiment

# In[21]:


RunDetails(automl_run).show()


# ## Best Model
# 

# Retrieve the best run

# In[22]:


best_run = package_utils.get_best_run(experiment, automl_run)


# In[23]:


print(best_run)
print(best_run.get_metrics())


# Save the best model locally

# In[24]:


print(best_run.get_details())


# In[25]:


print(best_run.get_properties())


# In[26]:


get_ipython().run_cell_magic('time', '', 'best_run.download_file("outputs/model.pkl", "models/automl/model.pkl")')


# In[27]:


print("model size in Bytes", pathlib.Path("models/automl/model.pkl").stat().st_size)


# Download other files from the best_run

# In[28]:


print(best_run.get_file_names())


# In[29]:


best_run.download_file("outputs/conda_env_v_1_0_0.yml",
                       "models/automl/conda_env_v_1_0_0.yml")
best_run.download_file("outputs/scoring_file_v_1_0_0.py",
                       "models/automl/scoring_file_v_1_0_0.py")
best_run.download_file("outputs/env_dependencies.json",
                       "models/automl/env_dependencies.json")

best_run.download_file("automl_driver.py",
                       "models/automl/automl_driver.py")
best_run.download_file("logs/azureml/azureml_automl.log",
                       "models/automl/azureml_automl.log")
best_run.download_file("azureml-logs/70_driver_log.txt",
                       "models/automl/70_driver_log.txt")


# Explore the model

# In[30]:


get_ipython().run_cell_magic('time', '', 'if 0:  # This method works only for AutoML\n    _, fitted_model = automl_run.get_output()')


# In[31]:


get_ipython().run_cell_magic('time', '', 'fitted_model = joblib.load("models/automl/model.pkl")')


# In[32]:


info = {}

for key, value in fitted_model.get_params()["Pipeline"].items():
    if key == "prefittedsoftvotingclassifier__weights":
        info[key] = value
    if key == "prefittedsoftvotingclassifier__estimators":
        info[key] = []
        for estimator in value:
            constitute_model = estimator[1].steps[-1][-1]
            info[key].append(constitute_model)

pprint.pprint(info)


# ## Model testing

# Test the model on the test set from 2018 and on an additional out-of-sample test set from 2017.

# Test the model on the 2018 dataset

# In[33]:


get_ipython().run_cell_magic('time', '', 'test = Dataset.get_by_name(\n            workspace=workspace,\n            name=dataset_test_name,\n        ).to_pandas_dataframe()\nx_test, y_test = test.drop(labels=["Label"], axis=1), test["Label"]')


# In[34]:


get_ipython().run_cell_magic('time', '', 'y_test_predict = fitted_model.predict(x_test)')


# In[35]:


get_ipython().run_cell_magic('time', '', 'cr = classification_report(digits=4,\n                           y_true=y_test,\n                           y_pred=y_test_predict,\n                           output_dict=False)\nprint(cr)')


# In[36]:


del test


# Test the model on the 2017 dataset

# In[50]:


get_ipython().run_cell_magic('time', '', 'test_2017 = Dataset.get_by_name(\n            workspace=workspace,\n            name=dataset_2017_name,\n        ).to_pandas_dataframe()\nx_test_2017, y_test_2017 = test_2017.drop(labels=["Label"], axis=1), test_2017["Label"]')


# In[51]:


get_ipython().run_cell_magic('time', '', 'y_test_2017_predict = fitted_model.predict(x_test_2017)')


# In[52]:


get_ipython().run_cell_magic('time', '', 'cr = classification_report(digits=4,\n                           y_true=y_test_2017,\n                           y_pred=y_test_2017_predict,\n                           output_dict=False)\nprint(cr)')


# In[53]:


del test_2017
del x_test_2017


# ## Model Deployment

# Register the best model into the workspace

# In[37]:


get_ipython().run_cell_magic('time', '', 'model = package_utils.register_model(model_name="automl-jupyter",\n                                     model_path="outputs/model.pkl",\n                                     run=best_run)')


# In[38]:


print(model.serialize())


# Deploy the registered model to an Azure Container instance

# In[39]:


get_ipython().run_cell_magic('time', '', 'service = deploy_config.main(model_name="automl-jupyter",\n                             deployment_name="automl-jupyter")')


# In[40]:


get_ipython().run_cell_magic('time', '', 'service.wait_for_deployment(show_output=True)')


# In[41]:


assert service.state == "Healthy"


# Test the service endpoint

# Fetch the API keys of the service endpoint

# In[42]:


primary_api_key, secondary_api_key = service.get_keys()


# Retrive the scoring url of the service endpoint

# In[43]:


url = service.scoring_uri
print(url)


# Prepare a subset of the test dataset for submission to the service

# In[44]:


input_data = json.dumps({'data': x_test[0:1].to_dict(orient='records')})
with open("data.json", "w") as _f:
    _f.write(input_data)
get_ipython().system('cat data.json')


# Call the service using the input_data

# In[45]:


print(service.run(input_data))


# Send a post request to the service endpoint using curl

# In[46]:


get_ipython().run_cell_magic('time', '', '!curl -X POST \\\n      -H \'Content-Type: application/json\' \\\n      -H "Authorization: Bearer $secondary_api_key" \\\n      --data @data.json $url')


# Send a post request to the service endpoint programatically

# In[47]:


# Set the content type
headers = {"Content-Type": "application/json"}
# If authentication is enabled, set the authorization header
headers["Authorization"] = f"Bearer {secondary_api_key}"

resp = requests.post(url, input_data, headers=headers)
print(resp.json())


# In[49]:


del x_test


# Print the service logs

# In[54]:


print(service.get_logs())


# Delete the service endpoint

# In[55]:


service.delete()


# Delete the compute cluster

# In[56]:


cluster_name = package_utils.trim_cluster_name(workspace.name)
print(f"cluster_name {cluster_name}")


# In[59]:


try:
    compute_target = ComputeTarget(workspace=workspace, name=cluster_name)
    compute_target.delete()
    print(f"compute_target {compute_target} is being deleted")    
except ComputeTargetException:
    print(f"compute_target {cluster_name} does not exist")


# In[ ]:




