import pytorch_lightning as pl
import math
from torch import optim
import torch
from transformers import AutoModelForCausalLM
from carbontracker.tracker import CarbonTracker
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


class GPT(pl.LightningModule):
    def __init__(self, model_name: str, len_tokenizer: int, lr: float=2e-5) -> object:
        super().__init__()
        self.tracker = None
        self.gpt = AutoModelForCausalLM.from_pretrained(model_name)
        self.gpt.resize_token_embeddings(len_tokenizer)
        self.lr = lr
        self.early_stopping = EarlyStopping(
            monitor="val_ppl", 
            min_delta=1, 
            patience=2, 
            verbose=False, 
            mode="min",
        )
        self.checkpoint_callback = ModelCheckpoint(
            dirpath="bloom_checkpoints/E3C_CAS/7b1",
            #dirpath="gpt-e3c_checkpoints/base_balises",
            monitor="val_ppl",
            #save_on_train_epoch_end=True,
            save_weights_only=True,
            save_top_k = 5,
            filename="e3c_{epoch:02d}_{val_ppl:.2f}"
	)
        self.trainer = None

    def forward(self, x):
        self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.gpt.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch["input_ids"], batch["labels"]
        pred = self.gpt(x, labels=y)

        loss = pred[0]
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch["input_ids"], batch["labels"]
        pred = self.gpt(x, labels=y)

        loss = pred[0]
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return {"loss": loss}

    def on_train_epoch_start(self):
        if self.tracker is not None:
            self.tracker.epoch_start()

    def training_epoch_end(self, outs):
        avg_train_loss = torch.tensor([x["loss"] for x in outs]).mean()
        train_ppl = math.exp(avg_train_loss)

        print(f"Epoch {self.current_epoch + 1} - Perplexity on train set : {train_ppl:2f}")

    def validation_epoch_end(self, outs):
        if not self.trainer.sanity_checking:
            if self.tracker is not None:
                self.tracker.epoch_end()
        avg_val_loss = torch.tensor([x["loss"] for x in outs]).mean()
        val_ppl = math.exp(avg_val_loss)

        print(f"Epoch {self.current_epoch + 1} - Perplexity on validation set : {val_ppl:2f}")
        self.log("val_ppl", val_ppl)

    def set_carbon_tracking(self, max_epochs, log_dir, monitor_epochs=-1):
        self.tracker = CarbonTracker(epochs=max_epochs, monitor_epochs=monitor_epochs, log_dir=log_dir)

    def generate(self, **kwargs):
        output = self.gpt.generate(**kwargs)
        return output

    def start_training(self, train_loader, val_loader, epochs):
        self.trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu", devices=1, callbacks=[self.early_stopping, self.checkpoint_callback])
        self.trainer.fit(self, train_loader, val_loader)
        print(f"Best model : {self.checkpoint_callback.best_model_path}")
        print(f"Perplexity on validation set : {self.checkpoint_callback.best_model_score}")
	
        
