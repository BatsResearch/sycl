# SyCL

Code for [Beyond Contrastive Learning: Synthetic Data Enables List-wise Training with Multiple Levels of Relevance](https://www.arxiv.org/abs/2503.23239).

**Data**: the synthetic datasets are available in this huggingface datasets repo: [BatsResearch/sycl](https://huggingface.co/datasets/BatsResearch/sycl)

## Experiments

**Install**

Install [Trove](https://github.com/BatsResearch/trove):

```bash
pip install ir-trove
```

Install deepspeed:

```bash
pip install deepspeed
```

If you encounter any problems during installation, please refer to [Trove](https://github.com/BatsResearch/trove) and [deepspeed](https://www.deepspeed.ai/tutorials/advanced-install) documentations.

Use the following commands for the four main training experiments in SyCL.


**Train**

Train with binary relevance labels using real data with InfoNCE loss:

```bash
deepspeed --include localhost:0,1,2,3 train.py \
    --deepspeed="deepspeed_conf.json" \
    --dataset_name='msmarco' \
    --mqrel_conf="data_configs/real_binary.json" \
    --model_name_or_path="facebook/contriever" \
    --encoder_class='default' \
    --pooling='mean' \
    --normalize='no' \
    --loss="infonce" \
    --trust_remote_code='true' \
    --group_size="3" \
    --query_max_len='256' \
    --passage_max_len='256' \
    --output_dir="./model_output/ft_on_real_binary" \
    --report_to='none' \
    --save_strategy='epoch' \
    --per_device_train_batch_size="16" \
    --learning_rate="1e-5" \
    --num_train_epochs="1" \
    --logging_steps='1' \
    --gradient_accumulation_steps='4' \
    --warmup_ratio='0.05' \
    --eval_strategy='no' \
    --dataloader_num_workers="2" \
    --save_only_model='true'
```

Train with binary relevance labels using synthetic data with InfoNCE loss:

```bash
deepspeed --include localhost:0,1,2,3 train.py \
    --deepspeed="deepspeed_conf.json" \
    --dataset_name='msmarco' \
    --mqrel_conf="data_configs/synth_binary.json" \
    --model_name_or_path="facebook/contriever" \
    --encoder_class='default' \
    --pooling='mean' \
    --normalize='no' \
    --loss="infonce" \
    --trust_remote_code='true' \
    --group_size="3" \
    --query_max_len='256' \
    --passage_max_len='256' \
    --output_dir="./model_output/ft_on_synth_binary" \
    --report_to='none' \
    --save_strategy='epoch' \
    --per_device_train_batch_size="16" \
    --learning_rate="1e-5" \
    --num_train_epochs="1" \
    --logging_steps='1' \
    --gradient_accumulation_steps='4' \
    --warmup_ratio='0.05' \
    --eval_strategy='no' \
    --dataloader_num_workers="2" \
    --save_only_model='true'
```

Train with graduated relevance labels using synthetic data with Wasserstein loss:

```bash
deepspeed --include localhost:0,1,2,3 train.py \
    --deepspeed="deepspeed_conf.json" \
    --dataset_name='msmarco' \
    --mqrel_conf="data_configs/synth_multilevel.json" \
    --model_name_or_path="facebook/contriever" \
    --encoder_class='default' \
    --pooling='mean' \
    --normalize='no' \
    --loss="ws2" \
    --trust_remote_code='true' \
    --group_size="4" \
    --passage_selection_strategy='random' \
    --query_max_len='256' \
    --passage_max_len='256' \
    --output_dir="./model_output/ft_sycl_synth" \
    --report_to='none' \
    --save_strategy='epoch' \
    --per_device_train_batch_size="16" \
    --learning_rate="1e-5" \
    --num_train_epochs="1" \
    --logging_steps='1' \
    --gradient_accumulation_steps='4' \
    --warmup_ratio='0.05' \
    --eval_strategy='no' \
    --dataloader_num_workers="2" \
    --save_only_model='true'
```

Train with graduated relevance labels using a combination of synthetic and real data with Wasserstein loss:

```bash
deepspeed --include localhost:0,1,2,3 train.py \
    --deepspeed="deepspeed_conf.json" \
    --dataset_name='msmarco' \
    --mqrel_conf="data_configs/synth_plus_real.json" \
    --model_name_or_path="facebook/contriever" \
    --encoder_class='default' \
    --pooling='mean' \
    --normalize='no' \
    --loss="ws2" \
    --trust_remote_code='true' \
    --group_size="7" \
    --passage_selection_strategy='random' \
    --query_max_len='256' \
    --passage_max_len='256' \
    --output_dir="./model_output/ft_sycl_synth_plus_real" \
    --report_to='none' \
    --save_strategy='epoch' \
    --per_device_train_batch_size="16" \
    --learning_rate="1e-5" \
    --num_train_epochs="1" \
    --logging_steps='1' \
    --gradient_accumulation_steps='4' \
    --warmup_ratio='0.05' \
    --eval_strategy='no' \
    --dataloader_num_workers="2" \
    --save_only_model='true'
```

**Evaluate**

The following snippet evaluates `facebook/contriever` on `scifact`.
Change the environment variables to evaluate different models on different datasets.

**Attention:** make sure to use a different `MODEL_EMB_CACHE` for each model checkpoint.
Otherwise, after your first evaluation run, the same cached embeddings are used for all subsequent evaluations on the same dataset.

```bash
DATASET='scifact'
MODEL_NAME='facebook/contriever'
MODEL_EMB_CACHE='./encoding_cache/base_contriever'
OUTPUT_DIR="eval_output/base_contriever/${DATASET}_results"

deepspeed --include localhost:0,1,2,3 eval.py \
    --eval_data_conf='data_configs/eval_data.json' \
    --model_name_or_path="${MODEL_NAME}" \
    --encoder_class='default' \
    --pooling='mean' \
    --normalize='no' \
    --dataset_name="${DATASET}" \
    --query_max_len='256' \
    --passage_max_len='256' \
    --output_dir="${OUTPUT_DIR}" \
    --per_device_matmul_batch_size='40960' \
    --precompute_corpus_embs='true' \
    --encoding_cache_dir="${MODEL_EMB_CACHE}" \
    --pbar_mode='local_main' \
    --print_mode='local_main' \
    --cleanup_temp_artifacts='false' \
    --per_device_eval_batch_size="128" \
    --dataloader_num_workers="4" \
    --report_to='none' \
    --broadcast_output='false' \
    --save_eval_topk_logits='false'
```
