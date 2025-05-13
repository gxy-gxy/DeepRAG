
cd DeepRAG/src


DIR=result/hotpotqa/0

echo
cat $DIR/model_id.txt


python3 evaluate_ans.py \
    --dir ${DIR} \
    --remote_url "http://0.0.0.0:8000/v1"

