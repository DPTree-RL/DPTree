WORK_DIR=/home/users/ntu/zihan021/scratch/PMTS-GRPO
LOCAL_DIR=/home/users/ntu/zihan021/scratch/PMTS-GRPO/data/multihop

## process multiple dataset search format train file
DATA=hotpotqa
python $WORK_DIR/scripts/data_process/qa_search_train_merge.py --local_dir $LOCAL_DIR --data_sources $DATA

## process multiple dataset search format test file
DATA=hotpotqa,2wikimultihopqa,musique,bamboogle
python $WORK_DIR/scripts/data_process/qa_search_test_merge.py --local_dir $LOCAL_DIR --data_sources $DATA
