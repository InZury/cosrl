import os
import wandb
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from cosrl.dataset import COD10K
from cosrl.models import SwinEncoder, SwinDecoder
from cosrl.utils import FocalTverskyLoss, IOU, get_edge_mask
from cosrl.utils.visualize import save_test_result

def train(version):
    wandb.init(
        project="COSRL",
        name=f"[Train] COSRL {version}",
        config={
            "epochs": 120,
            "image_size": 352,
            "batch_size": 64,
            "num_workers": 4,
            "weight_decay": 1e-4,
            "loss_lambda": 0.05,

            "edge_start_epoch": 40,
            "edge_rampage_epoch": 10,

            "bce_pos_weight": 5.0,

            "tversky":{
                "alpha": 0.3,
                "beta": 0.7,
                "gamma": 0.75,
                "epsilon": 1e-6
            },

            "metric":{
                "threshold": 0.5,
                "epsilon": 1e-6
            },

            "encoder":{
                "pretrained_model_name_or_path": "microsoft/swin-tiny-patch4-window7-224",
                "unfreeze_encoder_epochs": 40,
            },

            "decoder":{
                "in_channels": (768, 384, 192, 96),
                "hidden_channels": 256,
                "num_stages": 3,
                "num_heads": (8, 8, 16),
                "num_attention": (2, 2, 1),
                "window_size": 8,
                "mlp_ratio": 4.0,
                "qkv_bias": True,
                "qk_scale": None,
                "attention_drop": 0.1,
                "projection_drop": 0.1,
                "drop_path": 0.0,
                "learning_rate": 1e-4
            }
        }
    )

    config = wandb.config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Dataset Init
    train_path = os.path.join("data", "Train")
    test_path = os.path.join("data", "Test")

    train_dataset = COD10K(path=train_path, image_size=config.image_size, only_camo=True, is_train=True)
    test_dataset = COD10K(path=test_path, image_size=config.image_size, only_camo=True, is_train=False)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True
    )

    # Encoder Init
    encoder = SwinEncoder(pretrained_model_name_or_path=config.encoder.pretrained_model_name_or_path, output_hidden_states=True)
    encoder.to(device).eval()
    encoder.set_requires_grad()

    # Decoder Init
    decoder = SwinDecoder(
        in_channels=config.decoder.in_channels,
        hidden_channels=config.decoder.hidden_channels,
        input_resolution=config.image_size,
        num_stages=config.decoder.num_stages,
        num_heads=config.decoder.num_heads,
        num_attention=config.decoder.num_attention,
        window_size=config.decodder.window_size,
        mlp_ratio=config.decoder.mlp_ratio,
        qkv_bias=config.decoder.qkv_bias,
        qk_scale=config.decoder.qk_scale,
        attention_drop=config.decoder.attention_drop,
        projection_drop=config.decoder.projection_drop,
        drop_path=config.decoder.drop_path,
        activation=nn.GELU,
        norm=nn.LayerNorm
    )
    decoder.to(device).train()

    # Optimizer + Scheduling Init
    optimizer = AdamW([
        {"params": encoder.parameters(), "lr": 0.0},
        {"params": decoder.parameters(), "lr": config.decoder.learning_rate}
    ], weight_decay=config.weight_decay)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=config.epochs * len(train_loader))
    scaler = torch.amp.GradScaler(device="cuda")

    def edge_lambda(epochs):
        start = int(config.edge_start_epoch)
        rampage = int(config.edge_rampage_epoch)

        if epochs < start:
            return 0.0
        if rampage <= 0:
            return float(config.loss_lambda)

        step = min(1.0, (epochs - start + 1) / float(rampage))

        return float(config.loss_lambda) * step

    # Loss Init
    bce_pos_weight = torch.tensor([config.bce_pos_weight], device=device)
    bce_mask = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight)
    bce_boundary = nn.BCEWithLogitsLoss()
    tversky_loss = FocalTverskyLoss(
        alpha=config.tversky.alpha, beta=config.tversky.beta, gamma=config.tversky.gamma,
        epsilon=config.tversky.epsilon, from_logits=True
    )

    # Metric Init
    metric = IOU(threshold=config.metric.threshold, epsilon=config.metric.epsilon)

    best_iou = 0.0
    is_unfreeze = False

    # Training Init
    for epoch in range(config.epochs):
        if epoch > config.unfreeze_encoder_epochs:
            encoder.train()

            encoder.set_requires_grad(layers=(2, 3))

            if not is_unfreeze:
                optimizer.param_groups[1]["lr"] = config.encoder.learning_rate
                scheduler.base_lrs[1] = config.encoder.learning_rate
                is_unfreeze = True

        decoder.train()

        lambda_edge = edge_lambda(epoch)

        train_losses = []
        train_ious = []
        train_ious_pos = []

        test_losses = []
        test_ious = []
        test_iou_pos = []

        # Train
        train_pbar = tqdm(train_loader, total=len(train_loader), desc=f"[Train] Epoch {epoch + 1}/{config.epochs}")

        for image, gt in train_loader:
            image = image.to(device)
            gt = gt.to(device)

            optimizer.zero_grad(set_to_none=True)

            if epoch < config.unfreeze_encoder_epochs:
                with torch.no_grad():
                    hidden_states = encoder(image)
            else:
                hidden_states = encoder(image)

            with torch.amp.autocast(dtype=torch.float16, device_type="cuda"):
                mask_logits, boundary_logits = decoder(hidden_states)
                loss_total = bce_mask(mask_logits, gt) + tversky_loss(mask_logits, gt)

                if lambda_edge > 0.0:
                    edge_gt = get_edge_mask(gt=gt, kernel=3)
                    loss_edge = bce_boundary(boundary_logits, edge_gt)
                    loss = loss_total + lambda_edge * loss_edge
                else:
                    loss = loss_total

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            params = list(decoder.parameters()) + [param for param in encoder.parameters() if param.requires_grad]
            torch.nn.utils.clip_grad_norm_(parameters=params, max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            iou, iou_pos = metric.compute_iou_pos(mask_logits, gt)
            lr = optimizer.param_groups[0]["lr"]

            train_losses.append(loss.item())
            train_ious.append(iou.item())
            train_ious_pos.append(iou_pos.item())

            train_pbar.set_postfix(
                loss=f"{np.mean(train_losses):.4f}",
                iou=f"{np.mean(train_ious):.4f}",
                iou_pos=f"{np.mean(train_ious_pos):.4f}",
                lr=f"{lr:.2e}"
            )

        # Test
        if epoch > config.unfreeze_encoder_epochs:
            encoder.eval()

        decoder.eval()

        test_pbar = tqdm(test_loader, total=len(test_loader), desc=f"[Test]  Epoch {epoch + 1}/{config.epochs}")

        with torch.no_grad():
            for image, gt in test_loader:
                image = image.to(device)
                gt = gt.to(device)

                with torch.amp.autocast(dtype=torch.float16, device_type="cuda"):
                    hidden_states = encoder(image)
                    mask_logits, boundary_logits = decoder(hidden_states)

                    loss = bce_mask(mask_logits, gt) + tversky_loss(mask_logits, gt)

                iou, iou_pos = metric.compute_iou_pos(mask_logits, gt)

                test_losses.append(loss.item())
                test_ious.append(iou.item())
                test_iou_pos.append(iou_pos.item())

                test_pbar.set_postfix(
                    loss=f"{np.mean(test_losses):.4f}",
                    iou=f"{np.mean(test_ious):.4f}",
                    iou_pos=f"{np.mean(test_iou_pos):.4f}"
                )

        wandb.log({
            "train_loss": np.mean(train_losses),
            "test_loss": np.mean(test_losses),
            "train_iou": np.mean(train_ious),
            "test_iou": np.mean(test_ious),
            "train_iou_pos": np.mean(train_ious_pos),
            "test_iou_pos": np.mean(test_iou_pos),
            "edge_lambda": lambda_edge,
            "step": epoch + 1
        })

        if np.mean(test_ious) > best_iou:
            best_iou = np.mean(test_ious)

            os.makedirs("checkpoints", exist_ok=True)
            weight_path = os.path.join("checkpoints", f"cosrl_{version}")

            torch.save(encoder.state_dict(), f"{weight_path}_encoder.pth")
            torch.save(decoder.state_dict(), f"{weight_path}_decoder.pth")

    save_test_result(
        dataset=test_dataset, num_image=5, encoder=encoder, decoder=decoder, device=device, metric=metric, version=version
    )


if __name__ == '__main__':
    train(version="0.1.0")