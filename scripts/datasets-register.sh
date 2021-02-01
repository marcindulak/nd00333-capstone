dataset_train_name="train"
dataset_validate_name="validate"
dataset_test_name="test"

az ml datastore upload --name workspaceblobstore --verbose --src-path datasets/$dataset_train_name --target-path $dataset_train_name
az ml datastore upload --name workspaceblobstore --verbose --src-path datasets/$dataset_validate_name --target-path $dataset_validate_name
az ml datastore upload --name workspaceblobstore --verbose --src-path datasets/$dataset_test_name --target-path $dataset_test_name
PYTHONPATH=. python nd00333/dataset/register/register.py --dataset-path datasets --dataset-name $dataset_train_name --dataset-version 1 --dataset-type file
PYTHONPATH=. python nd00333/dataset/register/register.py --dataset-path datasets --dataset-name $$dataset_validate_name --dataset-version 1 --dataset-type file
PYTHONPATH=. python nd00333/dataset/register/register.py --dataset-path datasets --dataset-name $dataset_test_name --dataset-version 1 --dataset-type tabular
