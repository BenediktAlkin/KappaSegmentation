from ksuit.factory import MasterFactory
from ksuit.models import CompositeModel
from ksuit.models.poolings import ToImage


class EncoderDecoder(CompositeModel):
    def __init__(
            self,
            encoder,
            decoder,
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
        # decoder
        self.decoder = MasterFactory.get("model").create(
            decoder,
            input_shape=self.encoder.output_shape,
            output_shape=self.output_shape,
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
        )

    @property
    def submodels(self):
        return dict(encoder=self.encoder, decoder=self.decoder)

    # noinspection PyMethodOverriding
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def segment(self, x):
        return self.forward(x)
