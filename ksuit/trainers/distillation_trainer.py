import torch.nn.functional as F
from torch import nn

from ksuit.callbacks.online_callbacks import OnlineAccuracyCallback
from ksuit.factory import MasterFactory
from .base import SgdTrainer


class DistillationTrainer(SgdTrainer):
    def __init__(
            self,
            temperature,
            classification_loss_weight=1.0,
            logits_distillation_loss_weight=1.0,
            forward_kwargs=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.classification_loss_weight = classification_loss_weight
        self.logits_distillation_loss_weight = logits_distillation_loss_weight
        self.forward_kwargs = MasterFactory.create_dict(forward_kwargs)

    def get_trainer_callbacks(self, model=None):
        # select suited callback_ctor for dataset type (binary/multiclass/multilabel)
        ds = self.data_container.get_dataset("train")
        if ds.getdim("class") <= 2:
            raise NotImplementedError(f"binary classification not supported")
        # create callbacks
        return [
            OnlineAccuracyCallback(
                verbose=False,
                every_n_updates=self.track_every_n_updates,
                every_n_samples=self.track_every_n_samples,
                **self.get_default_callback_kwargs(),
            ),
            OnlineAccuracyCallback(
                **self.get_default_callback_intervals(),
                **self.get_default_callback_kwargs(),
            ),
        ]

    @property
    def output_shape(self):
        return self.data_container.get_dataset("train").getdim("class"),

    @property
    def dataset_mode(self):
        return "index x class"

    def get_trainer_model(self, model):
        return self.Model(model=model, trainer=self)

    class Model(nn.Module):
        def __init__(self, model, trainer):
            super().__init__()
            self.model = model
            self.trainer = trainer

        # noinspection PyDictCreation
        def forward(self, batch, reduction="mean"):
            # prepare data
            idx = batch["index"]
            x = batch["x"].to(self.model.device, non_blocking=True)
            target = batch["class"].to(self.model.device, non_blocking=True)

            # forward
            outputs = self.model(x)
            student_logits = outputs["student_logits"]
            teacher_logits = outputs["teacher_logits"]
            assert not teacher_logits.requires_grad

            # calculate losses
            losses = {}
            losses["supervised_loss"] = F.cross_entropy(student_logits, target, reduction=reduction)
            # https://github.com/megvii-research/mdistiller/blob/master/mdistiller/distillers/KD.py#L8
            student_lsm = F.log_softmax(student_logits / self.trainer.temperature, dim=-1)
            teacher_sm = F.softmax(teacher_logits / self.trainer.temperature, dim=-1)
            assert reduction == "mean"
            assert student_lsm.ndim == 2 and teacher_sm.ndim == 2
            losses["distill_loss"] = (
                    F.kl_div(student_lsm, teacher_sm, reduction="none").sum(dim=-1).mean()
                    * self.trainer.temperature ** 2
            )

            losses["total"] = (
                    losses["supervised_loss"] * self.trainer.classification_loss_weight
                    + losses["distill_loss"] * self.trainer.logits_distillation_loss_weight
            )

            # compose outputs (for callbacks to use)
            outputs = {
                "idx": idx,
                "preds": dict(
                    student=student_logits,
                    teacher=teacher_logits,
                ),
                "target": target,
            }
            return losses, outputs
