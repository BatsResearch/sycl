import json
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.linalg as linalg
import torch.nn.functional as F
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from trove import (
    BiEncoderRetriever,
    BinaryDataset,
    DataArguments,
    MaterializedQRelConfig,
    ModelArguments,
    MultiLevelDataset,
    RetrievalCollator,
    RetrievalLoss,
    RetrievalTrainer,
    RetrievalTrainingArguments,
)


class WS2Loss(RetrievalLoss):
    _alias = "ws2"

    def __init__(self, args: ModelArguments, **kwargs) -> None:
        """Implements 2-wasserstein loss."""
        super().__init__()

        if args.temperature_learnable:
            raise NotImplementedError
        self.temperature = float(args.temperature)

        self.b = -1
        self.fact = None
        self.fact_sqrt = None

    def calc_dist(self, X, Y):
        """
        Calulates the two components of the 2-Wasserstein metric:
        The general formula is given by: d(P_X, P_Y) = min_{X, Y} E[|X-Y|^2]

        For multivariate gaussian distributed inputs z_X ~ MN(mu_X, cov_X) and z_Y ~ MN(mu_Y, cov_Y),
        this reduces to: d = |mu_X - mu_Y|^2 - Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))

        Fast method implemented according to following paper: https://arxiv.org/pdf/2009.14075.pdf

        implementation from here: https://gist.github.com/Flunzmas/6e359b118b0730ab403753dcc2a447df.

        Input shape: [b, n] (e.g. batch_size x num_features)
        Output shape: scalar
        """
        # the linear algebra ops will need some extra precision -> convert to double
        X, Y = X.transpose(0, 1).double(), Y.transpose(0, 1).double()  # [n, b]
        mu_X, mu_Y = torch.mean(X, dim=1, keepdim=True), torch.mean(
            Y, dim=1, keepdim=True
        )  # [n, 1]

        _, b = X.shape
        if b != self.b:
            self.fact = 1.0 if b < 2 else 1.0 / (b - 1)
            self.fact_sqrt = math.sqrt(self.fact)
            self.b = b

        # Cov. Matrix
        E_X = X - mu_X
        E_Y = Y - mu_Y
        cov_X = torch.matmul(E_X, E_X.t()) * self.fact  # [n, n]
        cov_Y = torch.matmul(E_Y, E_Y.t()) * self.fact

        # calculate Tr((cov_X * cov_Y)^(1/2)). with the method proposed in https://arxiv.org/pdf/2009.14075.pdf
        # The eigenvalues for M are real-valued.
        C_X = E_X * self.fact_sqrt  # [n, n], "root" of covariance
        C_Y = E_Y * self.fact_sqrt
        M_l = torch.matmul(C_X.t(), C_Y)
        M_r = torch.matmul(C_Y.t(), C_X)
        M = torch.matmul(M_l, M_r)
        S = (
            linalg.eigvals(M) + 1e-15
        )  # add small constant to avoid infinite gradients from sqrt(0)
        sq_tr_cov = S.sqrt().abs().sum()

        # plug the sqrt_trace_component into Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
        trace_term = torch.trace(cov_X + cov_Y) - 2.0 * sq_tr_cov  # scalar

        # |mu_X - mu_Y|^2
        diff = mu_X - mu_Y  # [n, 1]
        mean_term = torch.sum(torch.mul(diff, diff))  # scalar

        # put it together
        return trace_term + mean_term

    def forward(
        self, logits: torch.Tensor, label: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Calculates the loss given the similarity scores between query and passages."""
        if label.size(1) != logits.size(1):
            label = torch.block_diag(*torch.chunk(label, label.shape[0]))

        preds = F.softmax(logits / self.temperature, dim=1)
        targets = F.softmax(label.double(), dim=1)
        loss = self.calc_dist(X=preds, Y=targets)
        return loss


@dataclass
class ScriptArguments:
    mqrel_conf: Optional[str] = None


def main():
    # parse arguments
    parser = HfArgumentParser(
        (RetrievalTrainingArguments, ModelArguments, DataArguments, ScriptArguments)
    )
    train_args, model_args, data_args, args = parser.parse_args_into_dataclasses()
    set_seed(train_args.seed)

    # create the model
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        clean_up_tokenization_spaces=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = BiEncoderRetriever.from_model_args(
        args=model_args, training_args=train_args
    )

    # create data objects
    with open(args.mqrel_conf, "r") as f:
        mqrel_args = json.load(f)

    mqrel_conf = dict()
    for k in ["pos_mqrel", "neg_mqrel", "mqrel"]:
        if k in mqrel_args:
            mqrel_conf[k] = [MaterializedQRelConfig(**c) for c in mqrel_args[k]]

    if "pos_mqrel" in mqrel_conf and "neg_mqrel" in mqrel_conf:
        train_dataset = BinaryDataset(
            data_args=data_args,
            positive_configs=mqrel_conf["pos_mqrel"],
            negative_configs=mqrel_conf["neg_mqrel"],
            format_query=model.format_query,
            format_passage=model.format_passage,
            num_proc=8,
        )
    elif "mqrel" in mqrel_conf:
        train_dataset = MultiLevelDataset(
            data_args=data_args,
            qrel_config=mqrel_conf["mqrel"],
            format_query=model.format_query,
            format_passage=model.format_passage,
            num_proc=8,
        )

    data_collator = RetrievalCollator(
        data_args=data_args,
        tokenizer=tokenizer,
        append_eos=model.append_eos_token,
    )

    # train
    trainer = RetrievalTrainer(
        args=train_args,
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
