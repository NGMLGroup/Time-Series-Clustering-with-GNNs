import torch
from tsl.engines import Predictor
from tsl.data import Data


class CustomPredictor(Predictor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.temp_step_size = (self.model.softmax_temp - 0.01) / 100

    def predict_batch(self, batch: Data,
                      preprocess: bool = False,
                      postprocess: bool = True,
                      return_target: bool = False,
                      get_latents: bool = False,
                      **forward_kwargs):
        """"""
        inputs, targets, mask, transform = self._unpack_batch(batch)
        if preprocess:
            for key, trans in transform.items():
                if key in inputs:
                    inputs[key] = trans.transform(inputs[key])

        if get_latents:
            with torch.no_grad():
                latents, s, _ = self.model.get_latent_factors(**inputs,
                                                              **forward_kwargs)
            return latents, s

        else:
            if forward_kwargs is None:
                forward_kwargs = dict()
            y_hat, *aux_loss = self.forward(**inputs, **forward_kwargs)
            # Rescale outputs
            if postprocess:
                trans = transform.get('y')
                if trans is not None:
                    y_hat = trans.inverse_transform(y_hat)
            if return_target:
                y = targets.get('y')
                return y, y_hat, mask, *aux_loss
            return y_hat, *aux_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """"""
        # Unpack batch
        x, y, mask, transform = self._unpack_batch(batch)

        # Make predictions
        y_hat, _ = self.predict_batch(batch, preprocess=False, postprocess=True)

        output = dict(**y, y_hat=y_hat)
        if mask is not None:
            output['mask'] = mask
        return output

    def training_step(self, batch, batch_idx):
        """"""
        y = y_loss = batch.y
        mask = batch.get('mask')

        # Compute predictions and compute loss
        y_hat_loss, *aux_loss = self.predict_batch(batch, preprocess=False,
                                        postprocess=not self.scale_target)
        y_hat = y_hat_loss.detach()

        # Scale target and output, eventually
        if self.scale_target:
            y_loss = batch.transform['y'].transform(y)
            y_hat = batch.transform['y'].inverse_transform(y_hat)

        # Compute loss
        output_loss = self.loss_fn(y_hat_loss, y_loss, mask=mask)
        loss = output_loss + sum(aux_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """"""
        y = y_loss = batch.y
        mask = batch.get('mask')

        # Compute predictions
        y_hat_loss, *aux_loss = self.predict_batch(batch, preprocess=False,
                                        postprocess=not self.scale_target)
        y_hat = y_hat_loss.detach()

        # Scale target and output, eventually
        if self.scale_target:
            y_loss = batch.transform['y'].transform(y)
            y_hat = batch.transform['y'].inverse_transform(y_hat)

        # Compute loss
        output_loss = self.loss_fn(y_hat_loss, y_loss, mask=mask)
        val_loss = output_loss + sum(aux_loss)

        return val_loss

    def test_step(self, batch, batch_idx):
        """"""
        # Compute outputs and rescale
        y_hat, *aux_loss = self.predict_batch(batch, preprocess=False,
                                              postprocess=True)

        y, mask = batch.y, batch.get('mask')

        test_loss = self.loss_fn(y_hat, y, mask) + sum(aux_loss)

        return test_loss

    def compute_metrics(self, batch, preprocess=False, postprocess=True):
        """"""
        # Compute outputs and rescale
        y_hat, _ = self.predict_batch(batch, preprocess, postprocess)
        y, mask = batch.y, batch.get('mask')
        self.test_metrics.update(y_hat.detach(), y, mask)
        metrics_dict = self.test_metrics.compute()
        self.test_metrics.reset()
        return metrics_dict, y_hat

    def on_train_epoch_end(self) -> None:
        # Decrease softmax temperature gradually
        self.model.softmax_temp = max(
                                    0.01,
                                    self.model.softmax_temp -
                                    self.temp_step_size)
