import einops
import einops
import torch

from examples.msmim.extractors import VitBlockExtractor
from ksuit.factory import MasterFactory
from ksuit.models import CompositeModel
from ksuit.models.extractors.finalizers import StackFinalizer
from ksuit.models.poolings import ToImage

class VitUpernet(CompositeModel):
    def __init__(
            self,
            encoder,
            postprocessor,
            decoder,
            auxiliary,
            concat_cls=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        # encoder
        self.encoder = MasterFactory.get("model").create(
            encoder,
            input_shape=self.input_shape,
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
        )
        if self.encoder.depth == 12:
            block_indices = [3, 5, 7, 11]
        elif self.encoder.depth == 24:
            block_indices = [7, 11, 15, 23]
        elif self.encoder.depth == 32:
            block_indices = [7, 15, 23, 31]
        else:
            raise NotImplementedError
        pooling = ToImage(concat_cls=concat_cls, static_ctx=self.static_ctx)
        self.encoder_extractor = VitBlockExtractor(
            block_indices=block_indices,
            pooling=pooling,
        )
        self.encoder_extractor.register_hooks(self.encoder)
        pooling_output_shape = pooling.get_output_shape(self.encoder.output_shape)
        # postprocessor
        self.postprocessor = MasterFactory.get("model").create(
            postprocessor,
            input_shape=pooling_output_shape,
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
        )
        # auxiliary
        self.auxiliary = MasterFactory.get("model").create(
            auxiliary,
            input_shape=pooling_output_shape,
            output_shape=self.output_shape,
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
        )
        # decoder
        self.decoder = MasterFactory.get("model").create(
            decoder,
            input_shape=pooling_output_shape,
            output_shape=self.output_shape,
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
        )

    @property
    def submodels(self):
        return dict(
            encoder=self.encoder,
            postprocessor=self.postprocessor,
            auxiliary=self.auxiliary,
            decoder=self.decoder,
        )

    # noinspection PyMethodOverriding
    def forward(self, x):
        # encoder
        with self.encoder_extractor:
            _ = self.encoder(x)
        features = self.encoder_extractor.extract()

        # postprocess (up/down sampling of different layers)
        features = self.postprocessor(features)

        # auxiliary
        auxiliary = self.auxiliary(features)

        # decoder
        decoded = self.decoder(features)

        return dict(auxiliary=auxiliary, decoder=decoded)

    def segment(self, x):
        return self.forward(x)["decoder"]