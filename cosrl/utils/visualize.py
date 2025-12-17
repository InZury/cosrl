import os
import torch

# Non-Visual Rendering Setting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_test_result(dataset, num_image, encoder, decoder, device, metric, version):
    for index in range(num_image):
        image, gt = dataset[index]
        origin_image, origin_gt = dataset.get_origin(index)

        image = image.unsqueeze(0).to(device)
        gt = gt.unsqueeze(0).to(device)

        with torch.no_grad():
            hidden_states = encoder(image)
            mask_logits, _ = decoder(hidden_states)

            iou = metric(mask_logits, gt)
            mask = (mask_logits.sigmoid() > 0.5).detach().cpu().numpy()

        plt.figure(figsize=(15, 6))

        plt.subplot(1, 3, 1)
        plt.imshow(origin_image)
        plt.axis("off")
        plt.title("Input")

        plt.subplot(1, 3, 2)
        plt.imshow(origin_gt, cmap="gray")
        plt.axis("off")
        plt.title("Ground Truth")

        plt.subplot(1, 3, 3)
        plt.imshow(mask, cmap="gray")
        plt.axis("off")
        plt.title(f"Result (IoU={iou:.3f})")

        save_path = os.path.join("output", f"result")
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, os.path.join(f"train_{version}", f"train_{version}_{index}.png")))
        plt.close()
