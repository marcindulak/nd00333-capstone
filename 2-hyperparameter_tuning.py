#!/usr/bin/env python
# coding: utf-8

# # Hyperparameter Tuning using HyperDrive

# In[1]:


import datetime
import json
import pathlib
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


from nd00333.model.hyperdrive.train import run_config as hd_train_config
from nd00333.model.deploy import run_config as deploy_config
from nd00333 import utils as package_utils


# In[4]:


print("azureml.core.VERSION", azureml.core.VERSION)


# ## Create an experiment
# 
# Create an experiment identified by the creation date. The purpose of identifying the experiments is to not mix the manually run experiments (using this jupyter notebook) with the experiments run using the deployment automation of the master git branch.
# 

# In[5]:


import logging
logging.basicConfig(level=logging.ERROR)


# In[6]:


#!az logout


# In[7]:


workspace = Workspace.from_config()
utcnow = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H-%m")
utcnow = "2021-01-29-09-01"
experiment_name = f"jupyter-{utcnow}"


# In[8]:


experiment = Experiment(workspace, experiment_name)
print(f"experiment_name {experiment_name}")


# ## Dataset
# 
# Load the dataset from local files on disk into the default AzureML workspace datastore.
# Store the versioned (by the "_1" suffix) train, validate and test sets under separate datastore paths. The uploading of large datasets tends to break with "The write operation timed out" when using the Python SDK and therefore the `az` is used. See https://docs.microsoft.com/en-us/answers/questions/43980/cannot-upload-local-files-to-azureml-datastore-pyt.html

# In[9]:


dataset_train_name = "ids2018train_1"
dataset_validate_name = "ids2018validate_1"
dataset_test_name = "ids2018test_1"
dataset_2017_name = "ids2017full_1"


# In[9]:


get_ipython().system('az login')


# In[10]:


get_ipython().system('az extension add --name azure-cli-ml --version 1.21.0')


# In[11]:


get_ipython().system('az --version')


# In[12]:


get_ipython().run_cell_magic('time', '', '!az ml datastore upload --name workspaceblobstore --verbose \\\n    --src-path datasets/ids2018train --target-path \\\n    $dataset_train_name')


# In[13]:


get_ipython().run_cell_magic('time', '', '!az ml datastore upload --name workspaceblobstore --verbose \\\n    --src-path datasets/ids2018validate --target-path \\\n    $dataset_validate_name')


# In[14]:


get_ipython().run_cell_magic('time', '', '!az ml datastore upload --name workspaceblobstore --verbose \\\n    --src-path datasets/ids2018test --target-path \\\n    $dataset_test_name')


# In[15]:


get_ipython().run_cell_magic('time', '', '!az ml datastore upload --name workspaceblobstore --verbose \\\n    --src-path datasets/ids2017full --target-path \\\n    $dataset_2017_name')


# Upload and register the uploaded datasets into AzureML workspace. In principle the above `az` commands should not be necessary.

# In[16]:


get_ipython().run_cell_magic('time', '', '!PYTHONPATH=. python nd00333/dataset/register/register.py \\\n    --dataset-path datasets --dataset-name ids2018train --dataset-version 1 \\\n    --dataset-type file')


# In[17]:


get_ipython().run_cell_magic('time', '', '!PYTHONPATH=. python nd00333/dataset/register/register.py \\\n    --dataset-path datasets --dataset-name ids2018validate --dataset-version 1 \\\n    --dataset-type file')


# In[18]:


get_ipython().run_cell_magic('time', '', '!PYTHONPATH=. python nd00333/dataset/register/register.py \\\n     --dataset-path datasets --dataset-name ids2018test --dataset-version 1 \\\n     --dataset-type tabular')


# In[19]:


get_ipython().run_cell_magic('time', '', '!PYTHONPATH=. python nd00333/dataset/register/register.py \\\n     --dataset-path datasets --dataset-name ids2017full --dataset-version 1 \\\n     --dataset-type tabular')


# ## HyperDrive Configuration
# 
# The hyperparameter tuning is performed using a grid search on the [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). The following hyperparameters are varied:
# -  the number of trees in the forest `n_estimators`, which affects the model generalization, however more trees will result in longer individual model training time,
# - the `criterion` affects the sensitivity of the model to the minority classes, which is important for imbalanced datasets, like the one used here,
# - the `max_depth` may result in overfitting if set too high, moreover it may considerably increase the size of the model (to e.g. several hundreds Mbytes).
# 
# The ` BanditPolicy` termination policy is set such that it allows the grid search to explore all hyperparameter values, and the `max_concurrent_runs` is set to the number of grid search runs. The individual model runs are performed multi-threaded using all cores available on the compute instance by setting `n_jobs=-1`. The number of `max_concurrent_runs` is set to the number of the nodes in the compute cluster.
# 
# See [nd00333/model/hyperdrive/train/run_config.py](nd00333/model/hyperdrive/train/run_config.py) for more details.
# 
# The `norm_macro_recall` is used as the performance metrics. See [README.md](README.md) for the rationale.

# Perform the configuration of the HyperDrive run, including setting up the remote AML compute cluster

# In[20]:


get_ipython().run_cell_magic('time', '', 'hyperdrive_run_config = hd_train_config.main(dataset_train_name=dataset_train_name,\n                                             dataset_validate_name=dataset_validate_name)')


# Submit the HyperDrive configuration to the compute cluster

# In[21]:


get_ipython().run_cell_magic('time', '', 'hyperdrive_run = experiment.submit(config=hyperdrive_run_config, show_output=False)')


# In[22]:


get_ipython().run_cell_magic('time', '', 'hyperdrive_run.wait_for_completion(show_output=True)')


# In[24]:


assert(hyperdrive_run.get_status() == "Completed")


# ## Run Details

# The widget below shows the first few best runs of the experiment

# In[25]:


RunDetails(hyperdrive_run).show()


# ## Best Model
# 

# Retrieve the best run

# In[26]:


best_run = package_utils.get_best_run(experiment, hyperdrive_run)


# In[27]:


print(best_run)
print(best_run.get_metrics())


# Save the best model locally

# In[28]:


get_ipython().run_cell_magic('time', '', 'best_run.download_file("outputs/model.pkl", "models/hyperdrive/model.pkl")')


# In[29]:


print("model size in Bytes", pathlib.Path("models/hyperdrive/model.pkl").stat().st_size)


# Download other files from the best_run

# In[30]:


print(best_run.get_file_names())


# In[31]:


best_run.download_file("azureml-logs/70_driver_log.txt",
                       "models/hyperdrive/70_driver_log.txt")


# Explore the model

# In[32]:


get_ipython().run_cell_magic('time', '', 'fitted_model = joblib.load("models/hyperdrive/model.pkl")')


# In[33]:


print(fitted_model)


# ## Model testing

# Test the model on the test set from 2018 and on an additional out-of-sample test set from 2017.

# Test the model on the 2018 dataset

# In[34]:


get_ipython().run_cell_magic('time', '', 'test = Dataset.get_by_name(\n            workspace=workspace,\n            name=dataset_test_name,\n        ).to_pandas_dataframe()\nx_test, y_test = test.drop(labels=["Label"], axis=1), test["Label"]')


# In[35]:


get_ipython().run_cell_magic('time', '', 'y_test_predict = fitted_model.predict(x_test)')


# In[36]:


get_ipython().run_cell_magic('time', '', 'cr = classification_report(digits=4,\n                           y_true=y_test,\n                           y_pred=y_test_predict,\n                           output_dict=False)\nprint(cr)')


# In[37]:


del test


# Test the model on the 2017 dataset

# In[38]:


get_ipython().run_cell_magic('time', '', 'test_2017 = Dataset.get_by_name(\n            workspace=workspace,\n            name=dataset_2017_name,\n        ).to_pandas_dataframe()\nx_test_2017, y_test_2017 = test_2017.drop(labels=["Label"], axis=1), test_2017["Label"]')


# In[39]:


get_ipython().run_cell_magic('time', '', 'y_test_2017_predict = fitted_model.predict(x_test_2017)')


# In[40]:


get_ipython().run_cell_magic('time', '', 'cr = classification_report(digits=4,\n                           y_true=y_test_2017,\n                           y_pred=y_test_2017_predict,\n                           output_dict=False)\nprint(cr)')


# In[41]:


del test_2017
del x_test_2017


# ## Model Deployment

# Register the best model into the workspace

# In[42]:


get_ipython().run_cell_magic('time', '', 'model = package_utils.register_model(model_name="hyperdrive-jupyter",\n                                     model_path="outputs/model.pkl",\n                                     run=best_run)')


# Deploy the registered model to an Azure Container instance

# In[43]:


get_ipython().run_cell_magic('time', '', 'service = deploy_config.main(model_name="hyperdrive-jupyter",\n                             deployment_name="hyperdrive-jupyter")')


# In[44]:


get_ipython().run_cell_magic('time', '', 'service.wait_for_deployment(show_output=True)')


# In[45]:


assert service.state == "Healthy"


# Test the service endpoint

# Fetch the API keys of the service endpoint

# In[46]:


primary_api_key, secondary_api_key = service.get_keys()


# Retrive the scoring url of the service endpoint

# In[47]:


url = service.scoring_uri
print(url)


# Prepare a subset of the test dataset for submission to the service

# In[48]:


input_data = json.dumps({'data': x_test[0:1].to_dict(orient='records')})
with open("data.json", "w") as _f:
    _f.write(input_data)
get_ipython().system('cat data.json')


# Call the service using the input_data

# In[49]:


print(service.run(input_data))


# Send a post request to the service endpoint using curl

# In[50]:


get_ipython().run_cell_magic('time', '', '!curl -X POST \\\n      -H \'Content-Type: application/json\' \\\n      -H "Authorization: Bearer $secondary_api_key" \\\n      --data @data.json $url')


# Send a post request to the service endpoint programatically

# In[51]:


# Set the content type
headers = {"Content-Type": "application/json"}
# If authentication is enabled, set the authorization header
headers["Authorization"] = f"Bearer {secondary_api_key}"

resp = requests.post(url, input_data, headers=headers)
print(resp.json())


# In[53]:


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


# In[57]:


try:
    compute_target = ComputeTarget(workspace=workspace, name=cluster_name)
    compute_target.delete()
    print(f"compute_target {compute_target} is being deleted")    
except ComputeTargetException:
    print(f"compute_target {cluster_name} does not exist")


# In[ ]:




