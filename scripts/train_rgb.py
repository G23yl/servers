import argparse
import os

import lightning as pl
import torch
from diffusers.optimization import get_scheduler
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger

import wandb
from diffsynth import ModelManager, WanVideoPipeline
from examples.depth_wan.robodataset import RoboDepth


def check_exits(path_list):
    for path in path_list:
        if isinstance(path, list):
            check_exits(path)
        else:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required model file not found: {path}")


class LightningDepthWanModel(pl.LightningModule):
    def __init__(
        self,
        model_path: str,
        learning_rate: float = 1e-6,
        lr_scheduler_name: str = "constant_with_warmup",
        lr_warmup_steps: int = 200,
        lr_max_training_steps: int = 5000,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
    ):
        super().__init__()
        self.lr_scheduler_name = lr_scheduler_name
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_max_training_steps = lr_max_training_steps
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.learning_rate = learning_rate
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.exists(model_path):
            manager_input_paths = [
                f"{model_path}/step-004700.ckpt",
                f"{model_path}/models_t5_umt5-xxl-enc-bf16.pth",
                f"{model_path}/Wan2.1_VAE.pth",
                f"{model_path}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            ]
            check_exits(manager_input_paths)
            model_manager.load_models(
                manager_input_paths[:-1], torch_dtype=torch.bfloat16
            )
            model_manager.load_models(
                [manager_input_paths[-1]],
                torch_dtype=torch.float32,
            )

        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()
        self.pipe.denoising_model().requires_grad_(True)

    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()

    def configure_optimizers(self):
        trainable_modules = filter(
            lambda p: p.requires_grad, self.pipe.denoising_model().parameters()
        )
        optimizer = torch.optim.AdamW(
            trainable_modules, lr=self.learning_rate, weight_decay=0.001
        )

        lr_scheduler = get_scheduler(
            self.lr_scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=self.lr_max_training_steps,
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        self.pipe.device = self.device

        prompt = batch["prompts"]
        # [B, C, F, H, W]
        first_image = batch["first_image"].to(self.pipe.device)
        video = batch["frames"].to(self.pipe.device)

        batch_size, channels, frames, H, W = video.shape

        prompt_embed = self.pipe.encode_prompt(prompt)["context"].to(
            dtype=self.pipe.torch_dtype, device=self.pipe.device
        )
        clip_feat = self.pipe.image_encoder.encode_image(
            first_image.transpose(1, 2)
        ).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

        # prepare mask
        msk = torch.ones(batch_size, 1, frames, H // 8, W // 8, device=self.pipe.device)
        msk[:, :, 1:] = 0
        msk = torch.concat(
            [torch.repeat_interleave(msk[:, :, 0:1], repeats=4, dim=2), msk[:, :, 1:]],
            dim=2,
        )
        msk = msk.view(batch_size, msk.shape[2] // 4, 4, H // 8, W // 8)
        # [B, 4, 13, h, w]
        msk = msk.transpose(1, 2)

        vae_input = torch.concat(
            [
                first_image,
                torch.zeros(
                    batch_size, channels, frames - 1, H, W, device=first_image.device
                ),
            ],
            dim=2,
        ).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        y = self.pipe.vae.encode(vae_input, device=self.pipe.device).to(
            dtype=self.pipe.torch_dtype, device=self.pipe.device
        )
        # [B, 16+4, 13, h, w]
        y = torch.concat([msk, y], dim=1).to(
            dtype=self.pipe.torch_dtype, device=self.pipe.device
        )

        latent = self.pipe.vae.encode(video, device=self.pipe.device).to(
            dtype=self.pipe.torch_dtype, device=self.pipe.device
        )
        noise = torch.randn_like(latent)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(
            dtype=self.pipe.torch_dtype, device=self.pipe.device
        )
        noisy_latents = self.pipe.scheduler.add_noise(latent, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latent, noise, timestep)

        noise_pred = self.pipe.denoising_model()(
            noisy_latents,
            timestep,
            prompt_embed,
            clip_feat,
            y,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
        )

        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        self.log("loss", loss, prog_bar=True)

        return loss

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        state_dict = self.pipe.denoising_model().state_dict()
        checkpoint.update(state_dict)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_path", type=str, help="Path to save model")
    parser.add_argument("--model_path", type=str)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument(
        "--wandb_key", type=str, default="", help="wandb key to use wandb"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default="", help="wandb entity name to use wandb"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="", help="wandb project name to use wandb"
    )
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=50, help="saving interval")
    parser.add_argument("--lr_scheduler_name", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=200)
    parser.add_argument("--lr_max_training_steps", type=int, default=5000)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    wandb_logger = None
    if args.use_wandb:
        for key in ["wandb_key", "wandb_entity", "wandb_project"]:
            if getattr(args, key) == "":
                raise ValueError(f"Need {key} to use wandb")
        wandb.login(key=args.wandb_key)
        wandb_logger = WandbLogger(
            name="wan_rgb",
            entity=args.wandb_entity,
            project=args.wandb_project,
            mode="online",
        )

    logger.info(f"Loading dataset from {args.data_path}")
    dataset = RoboDepth(args.data_path, is_train=True, max_num_frames=49)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=10,
        drop_last=True,
    )

    logger.info(f"Dataloader size: {len(dataloader)}")

    logger.info("Loading model")
    model = LightningDepthWanModel(
        model_path=args.model_path,
        learning_rate=args.lr,
        lr_scheduler_name=args.lr_scheduler_name,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_max_training_steps=args.lr_max_training_steps,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
    )

    callback = ModelCheckpoint(
        dirpath=args.output_path,
        filename="step-{step:06d}",
        save_top_k=-1,
        save_weights_only=True,
        every_n_train_steps=args.save_steps,
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy="auto",
        accumulate_grad_batches=args.gradient_accumulation,
        callbacks=[callback],
        logger=wandb_logger,
        log_every_n_steps=1,
    )

    logger.info("Start training...")
    trainer.fit(model, dataloader)


"""
TOKENIZERS_PARALLELISM=False torchrun --nproc_per_node=1 \
    train.py \
    --data_path /mnt/zhouxin-mnt \
    --batch_size 1 \
    --lr 1e-5 \
    --output_path /mnt/zhouxin-mnt/self_forcing/rgb_ckpt \
    --model_path /mnt/zhouxin-mnt/ckpt/SkyReels-V2-I2V-1.3B-540P \
    --use_gradient_checkpointing \
    --max_epochs 1200 \
    --save_steps 50 \
    --lr_scheduler_name constant_with_warmup \
    --lr_warmup_steps 200 \
    --lr_max_training_steps 5000

TOKENIZERS_PARALLELISM=False torchrun --nproc_per_node=8 \
    train.py \
    --data_path /mnt/zhouxin-mnt \
    --batch_size 1 \
    --lr 2e-6 \
    --output_path /mnt/zhouxin-mnt/self_forcing/rgb_ckpt \
    --model_path /mnt/zhouxin-mnt/ckpt/SkyReels-V2-I2V-1.3B-540P \
    --use_wandb \
    --wandb_key e6e8b0991bb2913baef65b8c03438cc5a1ddff83 \
    --wandb_entity 1265126315-huazhong-university-of-science-and-technology \
    --wandb_project wan \
    --use_gradient_checkpointing \
    --max_epochs 5 \
    --save_steps 100 \
    --lr_scheduler_name constant_with_warmup \
    --lr_warmup_steps 1000 \
    --lr_max_training_steps 5000
"""
