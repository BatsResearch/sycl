import json
from dataclasses import dataclass
from typing import Optional

from transformers import AutoTokenizer, HfArgumentParser
from trove import (
    BiEncoderRetriever,
    DataArguments,
    EvaluationArguments,
    MaterializedQRelConfig,
    ModelArguments,
    MultiLevelDataset,
    RetrievalCollator,
    RetrievalEvaluator,
)


@dataclass
class ScriptArguments:
    eval_data_conf: Optional[str] = None


def main():
    parser = HfArgumentParser(
        (EvaluationArguments, ModelArguments, DataArguments, ScriptArguments)
    )
    eval_args, model_args, data_args, args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        clean_up_tokenization_spaces=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    model = BiEncoderRetriever.from_model_args(args=model_args)

    with open(args.eval_data_conf, "r") as f:
        qrel_conf = json.load(f)
        qrel_conf = qrel_conf[data_args.dataset_name]

    if data_args.dataset_name != "msmarco":
        qrel_conf = {"split": qrel_conf}

    eval_dataset = dict()
    for split, conf in qrel_conf.items():
        eval_dataset[split] = MultiLevelDataset(
            data_args=data_args,
            qrel_config=MaterializedQRelConfig(**conf),
            format_query=model.format_query,
            format_passage=model.format_passage,
            num_proc=8,
        )
    if data_args.dataset_name != "msmarco":
        eval_dataset = eval_dataset["split"]

    data_collator = RetrievalCollator(
        data_args=data_args,
        tokenizer=tokenizer,
        append_eos=model.append_eos_token,
    )
    evaluator = RetrievalEvaluator(
        args=eval_args,
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        eval_dataset=eval_dataset,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()
