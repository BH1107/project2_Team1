import numpy as np
import nni

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L


class CostomerModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        configs: dict,
    ):
        super().__init__()
        self.model = model
        self.configs = configs
        self.learning_rate = configs.get('learning_rate')

        self.val_losses = []
        self.test_losses = []
        self.test_accuracies = []

    def training_step(self, batch, batch_idx):
        X = batch.get('X')
        y = batch.get('y')
        y = y.squeeze()

        output = self.model(X).squeeze()  # 출력 차원 조정
        self.loss = F.binary_cross_entropy_with_logits(output, y)

        return self.loss

    def on_train_epoch_end(self, *args, **kwargs):
        self.log_dict(
            {'loss/train_loss': self.loss},
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.val_losses.clear()

        X = batch.get('X')
        y = batch.get('y')
        y = y.squeeze()

        output = self.model(X).squeeze()  # 출력 차원 조정
        self.val_loss = F.binary_cross_entropy_with_logits(output, y)
        self.val_losses.append(self.val_loss.item())

        return self.val_loss

    def on_validation_epoch_end(self):
        self.log_dict(
            {'loss/val_loss': self.val_loss,
             'learning_rate': self.learning_rate},
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        if self.configs.get('nni'):
            nni.report_intermediate_result(np.mean(self.val_losses))

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_losses.clear()
            self.test_accuracies.clear()

        X = batch.get('X')
        y = batch.get('y')
        y = y.squeeze()

        output = self.model(X).squeeze()
        test_loss = F.binary_cross_entropy_with_logits(output, y)  # 손실 계산
        self.test_losses.append(test_loss.item())

        predictions = (output > 0.5).float()
        accuracy = (predictions == y).float().mean()
        self.test_accuracies.append(accuracy.item())

        return {'test_loss': test_loss, 'test_accuracy': accuracy}

    def on_test_epoch_end(self):
        avg_test_loss = np.mean(self.test_losses)
        avg_test_accuracy = np.mean(self.test_accuracies)

        self.log('test_loss', avg_test_loss, prog_bar=True)
        self.log('test_accuracy', avg_test_accuracy, prog_bar=True)

        print(f'Final Test Loss: {avg_test_loss}')
        print(f'Final Test Accuracy: {avg_test_accuracy}')

        if self.configs.get('nni'):
            nni.report_final_result(avg_test_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'loss/val_loss',
        }
