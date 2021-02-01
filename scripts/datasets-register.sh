dataset_name="ids2018"
dataset_version="1"
dataset_train_name="${dataset_name}train"
dataset_validate_name="${dataset_name}validate"
dataset_test_name="${dataset_name}test"

time az ml datastore upload --name workspaceblobstore --verbose --src-path datasets/$dataset_train_name --target-path $dataset_train_name_${dataset_version}
time az ml datastore upload --name workspaceblobstore --verbose --src-path datasets/$dataset_validate_name --target-path $dataset_validate_name_${dataset_version}
time az ml datastore upload --name workspaceblobstore --verbose --src-path datasets/$dataset_test_name --target-path $dataset_test_name_${dataset_version}
PYTHONPATH=. time python nd00333/dataset/register/register.py --dataset-path datasets --dataset-name $dataset_train_name --dataset-version 1 --dataset-type file
PYTHONPATH=. time python nd00333/dataset/register/register.py --dataset-path datasets --dataset-name $$dataset_validate_name --dataset-version 1 --dataset-type file
PYTHONPATH=. time python nd00333/dataset/register/register.py --dataset-path datasets --dataset-name $dataset_test_name --dataset-version 1 --dataset-type tabular
