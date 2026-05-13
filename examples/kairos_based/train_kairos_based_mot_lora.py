import argparse
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random

import numpy as np
import torch
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from loguru import logger
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from examples.kairos_based.kairos_dataset import KairosMultiModalDataset, DummyKairosDataset
from examples.kairos_based.modules.kairos_model_modal_t2v import KairosMotModel
from examples.kairos_based.modules.text_encoders import QwenVLTextEncoder
from examples.kairos_based.modules.utils import init_weights_on_device, load_state_dict
from examples.kairos_based.modules.vaes import WanVideoVAE


DEFAULT_LORA_TARGET_MODULES = [
    "self_attn.q",
    "self_attn.k",
    "self_attn.v",
    "self_attn.o",
    "cross_attn.q",
    "cross_attn.k",
    "cross_attn.v",
    "cross_attn.o",
    "ffn.0",
    "ffn.2",
    "head.head",
]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_path", type=str, help="Path to save model")
    parser.add_argument("--model_path", type=str)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
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
    parser.add_argument("--save_steps", type=int, default=50, help="saving interval")
    parser.add_argument("--lr_scheduler_name", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=200)
    parser.add_argument("--max_training_steps", type=int, default=5000)
    parser.add_argument("--max_norm", type=float, default=10.0)
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to a saved LoRA adapter directory to resume from",
    )
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_bias",
        type=str,
        default="none",
        choices=["none", "all", "lora_only"],
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default=",".join(DEFAULT_LORA_TARGET_MODULES),
        help="Comma separated target module suffixes, or PEFT value such as all-linear",
    )
    parser.add_argument(
        "--lora_modules_to_save",
        type=str,
        default="",
        help="Comma separated extra modules to keep trainable and save with LoRA",
    )
    args = parser.parse_args()
    return args


def parse_csv_arg(value):
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_lora_target_modules(value):
    target_modules = parse_csv_arg(value)
    if target_modules is None:
        return DEFAULT_LORA_TARGET_MODULES
    if len(target_modules) == 1 and target_modules[0] == "all-linear":
        return "all-linear"
    return target_modules


def build_lora_config(args):
    return LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=parse_lora_target_modules(args.lora_target_modules),
        modules_to_save=parse_csv_arg(args.lora_modules_to_save),
    )


def load_base_model(args, ckpt_path_dict, accelerator, weight_type):
    model = KairosMotModel(
        device=accelerator.device,
        torch_dtype=weight_type,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
    )
    model.from_pretrained(ckpt_path_dict)
    model.requires_grad_(False)
    return model


def main():
    args = get_args()
    setup_seed(923)
    accelerator = Accelerator(gradient_accumulation_steps=4)

    weight_type = torch.float32
    if accelerator.mixed_precision == "bf16":
        weight_type = torch.bfloat16
    elif accelerator.mixed_precision == "fp16":
        weight_type = torch.float16

    if accelerator.is_local_main_process:
        logger.info(f"Using mixed precision: {weight_type}")
        logger.info(
            "Accelerator gradient accumulation steps: "
            f"{accelerator.gradient_accumulation_steps}"
        )

    ## set up wandb for logger
    if args.use_wandb and accelerator.is_local_main_process:
        for key in ["wandb_key", "wandb_entity", "wandb_project"]:
            if getattr(args, key) == "":
                raise ValueError(f"Need {key} to use wandb")
        wandb.login(key=args.wandb_key)
        wandb.init(
            mode="online",
            name="kairos_mot_lora",
            entity=args.wandb_entity,
            project=args.wandb_project,
        )

    ckpt_path_dict = {
        "dit": f"{args.model_path}/models/robot/kairos-robot-4B-480P-16fps.safetensors",
        "text_encoder": f"{args.model_path}/Qwen2.5-VL-7B-Instruct-AWQ/",
        "vae": f"{args.model_path}/Wan2.1-T2V-14B/Wan2.1_VAE.pth",
    }

    base_model = load_base_model(args, ckpt_path_dict, accelerator, weight_type)
    if args.resume:
        model = PeftModel.from_pretrained(
            base_model,
            args.resume,
            is_trainable=True,
        )
        if accelerator.is_local_main_process:
            logger.info(f"Loaded LoRA adapter from: {args.resume}")
    else:
        lora_config = build_lora_config(args)
        model = get_peft_model(base_model, lora_config)
        if accelerator.is_local_main_process:
            logger.info(
                f"Created LoRA adapter with targets: {lora_config.target_modules}"
            )

    if accelerator.is_local_main_process:
        model.print_trainable_parameters()

    # text_encoder
    text_encoder = QwenVLTextEncoder(
        dtype=weight_type,
        device=accelerator.device,
        from_pretrained=ckpt_path_dict["text_encoder"],
    )
    text_encoder.requires_grad_(False)
    # vae
    vae_state_dict = load_state_dict(ckpt_path_dict["vae"])
    vae_state_dict_converter = WanVideoVAE.state_dict_converter()
    state_dict_results = vae_state_dict_converter.from_civitai(vae_state_dict)
    extra_kwargs = {}
    with init_weights_on_device():
        vae = WanVideoVAE(**extra_kwargs)
    if hasattr(vae, "eval"):
        vae = vae.eval()
    vae.load_state_dict(state_dict_results, assign=True)
    vae = vae.to(dtype=weight_type, device=accelerator.device)
    vae.requires_grad_(False)
    if accelerator.is_local_main_process:
        logger.info("Loaded text_encoder and vae")

    parameters_optim = [p for p in model.parameters() if p.requires_grad]
    if len(parameters_optim) == 0:
        raise ValueError(
            "No trainable LoRA parameters found. Check --lora_target_modules."
        )
    # with open("trainable_params.txt", "w") as f:
    #     trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
    #     f.write("\n".join(trainable_params))
    # num_params = sum(p.numel() for p in parameters_optim)
    # if accelerator.is_local_main_process:
    #     logger.info(f"Number of trainable LoRA parameters: {num_params / 1e6:.2f}M")

    # prepare dataset
    dataset = KairosMultiModalDataset(
        data_root=args.data_path,
        modals=["depth", "flow"],
        samples_path="final_samples.json",
    )
    # dataset = DummyKairosDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )
    if accelerator.is_local_main_process:
        logger.info("Loaded dataset")

    ## optimizer and lr_scheduler
    optimizer = torch.optim.AdamW(parameters_optim, lr=args.lr, weight_decay=0.001)
    lr_scheduler = get_scheduler(
        args.lr_scheduler_name,
        optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_training_steps * accelerator.num_processes,
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    logger.info(
        f"[rank{accelerator.process_index}] dataloader length: {len(dataloader)}"
    )

    progress_bar = tqdm(
        range(0, args.max_training_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    global_step = 0
    while True:
        model.train()
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                video_prompt = batch["prompt"]
                modal_prompt = batch["modal_prompt"]
                first_image = batch["first_image"]
                video = batch["video"].to(dtype=weight_type)
                modal = batch["modal"].to(dtype=weight_type)

                ret = model(
                    video_prompt,
                    modal_prompt,
                    first_image,
                    video,
                    modal,
                    text_encoder,
                    vae,
                    return_dict=True,
                )
                video_loss, modal_loss, total_loss = (
                    ret["video_loss"],
                    ret["modal_loss"],
                    ret["total_loss"],
                )
                ## gather rgb_loss, modal_loss, loss between all processes
                avg_video_loss = accelerator.gather(video_loss).mean()
                avg_modal_loss = accelerator.gather(modal_loss).mean()
                avg_total_loss = accelerator.gather(total_loss).mean()

                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(parameters_optim, args.max_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            progress_bar.update(1)

            ## log
            if accelerator.is_local_main_process and args.use_wandb:
                wandb.log(
                    {
                        "total_loss": avg_total_loss.item(),
                        "video_loss": avg_video_loss.item(),
                        "modal_loss": avg_modal_loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )

            ## save ckpt
            if global_step % args.save_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_local_main_process:
                    out_dir = os.path.join(
                        args.output_path, f"step_{global_step:08d}"
                    )
                    os.makedirs(out_dir, exist_ok=True)
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        out_dir,
                        safe_serialization=True,
                    )
                    logger.info(f"LoRA adapter saved to {out_dir}")

            if global_step >= args.max_training_steps:
                break
        if global_step >= args.max_training_steps:
            break

    accelerator.end_training()


if __name__ == "__main__":
    main()
