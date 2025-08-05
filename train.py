import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.MSFA_Net import Generator_T   # Teacher model
from models.LI_Net import Generator, Discriminator   # Student model
from models.DAM import DAM
from models.convnext import convnext_small

from dataset import InpaintDataset
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import cv2
import os
from tqdm import tqdm
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import csv


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
seed_everything(42)


def feature_l1_loss(fake_images, gt_images, model, device):
    model.eval()
    with torch.no_grad():
        fake_features = model(fake_images.to(device))
        gt_features = model(gt_images.to(device))
    return nn.MSELoss()(fake_features, gt_features)


def train_gan_epoch(generator, generator_T, discriminator, adaptor_enc2, adaptor_bottleneck, adaptor_dec2, dataloader, criterion,
                    optimizer_g, optimizer_d, device, recognition_model, lambda_adv=0.1):
    generator.train()
    generator_T.eval()
    discriminator.train()
    epoch_g_loss, epoch_g_l2_loss, epoch_g_adv_loss, epoch_d_loss, epoch_g_recog_loss, epoch_kd_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    total_samples = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)


    for inputs, gts, masks, _ ,_ ,largemasks in progress_bar:
        batch_size = inputs.size(0)
        total_samples += batch_size
        inputs, gts, masks, largemasks = inputs.to(device), gts.to(device), masks.to(device), largemasks.to(device)

        optimizer_d.zero_grad()
        fake_images, student_enc2, student_bottleneck, student_dec2, student_dec1 = generator(inputs)
        real_output = discriminator(gts)
        fake_output = discriminator(fake_images.detach())

        d_loss_real = criterion(real_output, torch.ones_like(real_output).to(device))
        d_loss_fake = criterion(fake_output, torch.zeros_like(fake_output).to(device))
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()
        optimizer_g.zero_grad()


        with torch.no_grad():
            teacher_output, teacher_enc2, teacher_bottleneck, teacher_dec2, teacher_dec1 = generator_T(inputs)

        fake_output = discriminator(fake_images)
        g_loss_adv = criterion(fake_output, torch.ones_like(fake_output).to(device))
        g_loss_pixel = nn.MSELoss()(fake_images * (1 - largemasks), gts * (1 - largemasks)) + 100 * nn.MSELoss()(fake_images * largemasks, gts * largemasks)
        g_loss_recog = feature_l1_loss(fake_images, gts, recognition_model, device)

        kd_loss_output = nn.MSELoss()(fake_images * (1 - largemasks), teacher_output * (1 - largemasks)) + 100 * nn.MSELoss()(fake_images * largemasks,
                                                                                                                              teacher_output * largemasks)
        kd_loss_enc2 = nn.MSELoss()(adaptor_enc2(student_enc2), teacher_enc2)
        kd_loss_bottleneck = nn.MSELoss()(adaptor_bottleneck(student_bottleneck), teacher_bottleneck)
        kd_loss_dec2 = nn.MSELoss()(adaptor_dec2(student_dec2), teacher_dec2)
        kd_loss_dec1 = nn.MSELoss()(student_dec1, teacher_dec1)

        total_kd_loss = 10 * kd_loss_output + kd_loss_enc2 + kd_loss_bottleneck + kd_loss_dec2 + kd_loss_dec1

        g_loss = g_loss_pixel + lambda_adv * g_loss_adv + lambda_adv *g_loss_recog + 5*total_kd_loss
        g_loss.backward()
        optimizer_g.step()

        epoch_g_loss += g_loss.item()* batch_size
        epoch_g_l2_loss += g_loss_pixel.item()* batch_size
        epoch_g_adv_loss += g_loss_adv.item()* batch_size
        epoch_g_recog_loss += g_loss_recog.item() * batch_size
        epoch_d_loss += d_loss.item()* batch_size
        epoch_kd_loss += total_kd_loss.item()* batch_size

        progress_bar.set_postfix({"G_Loss": g_loss.item(), "G_L2_Loss": g_loss_pixel.item(), "G_adv_loss": g_loss_adv.item(), "G_recog_loss": g_loss_recog.item(),
                                  "D_Loss": d_loss.item(), "KD_Loss": total_kd_loss.item()})

    return (epoch_g_loss / total_samples,
            epoch_g_l2_loss / total_samples,
            epoch_g_adv_loss / total_samples,
            epoch_g_recog_loss / total_samples,
            epoch_d_loss / total_samples,
            epoch_kd_loss / total_samples)


def validate_epoch(generator, generator_T, discriminator, adaptor_enc2, adaptor_bottleneck, adaptor_dec2, dataloader, device, criterion, writer,
                   recognition_model, lambda_adv, epoch,save_dir=None):
    generator.eval()
    generator_T.eval()
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)

    val_g_loss, val_g_l2_loss, val_g_adv_loss, val_g_recog_loss, val_kd_loss = 0.0, 0.0, 0.0, 0.0, 0.0
    val_d_loss= 0.0
    total_samples = 0
    epoch_save_dir = os.path.join(save_dir, f'epoch_{epoch + 1}')
    os.makedirs(epoch_save_dir, exist_ok=True)


    with torch.no_grad():
        for i, (inputs, gts, masks, filenames, _, largemasks) in enumerate(dataloader):
            batch_size = inputs.size(0)
            total_samples += batch_size
            inputs, gts, masks, largemasks = inputs.to(device), gts.to(device), masks.to(device), largemasks.to(device)
            fake_images, student_enc2, student_bottleneck, student_dec2, student_dec1 = generator(inputs)
            teacher_output, teacher_enc2, teacher_bottleneck, teacher_dec2, teacher_dec1 = generator_T(inputs)

            if i < 6:
                images = inputs[:, :3, :, :]

                sample_images = fake_images.clamp(0, 1).cpu().numpy()
                gt_images = gts.clamp(0, 1).cpu().numpy()
                input_images = images.clamp(0,1).cpu().numpy()

                for idx in range(sample_images.shape[0]):
                    filename = filenames[idx]

                    sample_image_np = (sample_images[idx].transpose(1, 2, 0) * 255).astype(np.uint8)
                    gt_image_np = (gt_images[idx].transpose(1, 2, 0) * 255).astype(np.uint8)
                    input_image_np = (input_images[idx].transpose(1, 2, 0) * 255).astype(np.uint8)

                    cv2.imwrite(os.path.join(epoch_save_dir, f'sample_{i + 1}_{idx + 1}_{filename}.png'),
                                cv2.cvtColor(sample_image_np, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(epoch_save_dir, f'gt_{i + 1}_{idx + 1}_{filename}.png'),
                                cv2.cvtColor(gt_image_np, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(epoch_save_dir, f'input_{i + 1}_{idx + 1}_{filename}.png'),
                                cv2.cvtColor(input_image_np, cv2.COLOR_RGB2BGR))

            g_loss_adv = criterion(discriminator(fake_images),
                                   torch.ones_like(discriminator(fake_images)).to(device))
            g_loss_pixel = nn.MSELoss()(fake_images * (1 - largemasks), gts * (1 - largemasks)) + 100 * nn.MSELoss()(fake_images * largemasks, gts * largemasks)

            g_loss_recog = feature_l1_loss(fake_images, gts, recognition_model, device)

            kd_loss_output = nn.MSELoss()(fake_images * (1 - largemasks), teacher_output * (1 - largemasks)) + 100 * nn.MSELoss()(fake_images * largemasks,
                                                                                                                                  teacher_output * largemasks)
            kd_loss_enc2 = nn.MSELoss()(adaptor_enc2(student_enc2), teacher_enc2)
            kd_loss_bottleneck = nn.MSELoss()(adaptor_bottleneck(student_bottleneck), teacher_bottleneck)
            kd_loss_dec2 = nn.MSELoss()(adaptor_dec2(student_dec2), teacher_dec2)
            kd_loss_dec1 = nn.MSELoss()(student_dec1, teacher_dec1)
            total_kd_loss = 10 * kd_loss_output + kd_loss_enc2 + kd_loss_bottleneck + kd_loss_dec2 + kd_loss_dec1

            g_loss = g_loss_pixel + lambda_adv * g_loss_adv + lambda_adv * g_loss_recog + 5*total_kd_loss

            real_output = discriminator(gts)
            fake_output = discriminator(fake_images.detach())
            d_loss_real = criterion(real_output, torch.ones_like(real_output).to(
                device))
            d_loss_fake = criterion(fake_output, torch.zeros_like(fake_output).to(
                device))
            d_loss = d_loss_real + d_loss_fake


            val_g_loss += g_loss.item()* batch_size
            val_g_l2_loss += g_loss_pixel.item()* batch_size
            val_g_adv_loss += g_loss_adv.item()* batch_size
            val_g_recog_loss += g_loss_recog.item() * batch_size
            val_d_loss += d_loss.item()* batch_size
            val_kd_loss += total_kd_loss.item()* batch_size
            psnr(fake_images, gts)
            ssim(fake_images, gts)

    val_g_loss /= total_samples
    val_g_l2_loss /= total_samples
    val_g_adv_loss /= total_samples
    val_g_recog_loss /= total_samples
    val_d_loss /= total_samples
    val_kd_loss /= total_samples
    psnr_value = psnr.compute().item()
    ssim_value = ssim.compute().item()

    writer.add_scalar("Validation/G_Loss", val_g_loss, epoch)
    writer.add_scalar("Validation/G_L2_Loss", val_g_l2_loss, epoch)
    writer.add_scalar("Validation/G_adv_Loss", val_g_adv_loss, epoch)
    writer.add_scalar("Validation/G_recog_Loss", val_g_recog_loss, epoch)
    writer.add_scalar("Validation/D_Loss", val_d_loss, epoch)
    writer.add_scalar("Validation/KD_Loss", val_kd_loss, epoch)
    writer.add_scalar("Validation/PSNR", psnr_value, epoch)
    writer.add_scalar("Validation/SSIM", ssim_value, epoch)


    return val_g_loss, val_g_l2_loss, val_g_adv_loss, val_g_recog_loss, val_d_loss, val_kd_loss, psnr_value, ssim_value
def load_checkpoint(checkpoint_path, generator, discriminator, optimizer_g, optimizer_d):
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator1_state_dict"])
    optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
    optimizer_d.load_state_dict(checkpoint["optimizer_d1_state_dict"])
    epoch = checkpoint["epoch"]
    g_loss = checkpoint["g_loss"]
    g_l2_loss = checkpoint["g_l2_loss"]
    g_adv_loss = checkpoint["g_adv_loss"]
    d_loss = checkpoint["d_loss"]
    return generator, discriminator, optimizer_g, optimizer_d, epoch, g_loss, g_l2_loss, g_adv_loss, d_loss

def main():
    # Paths
    save_dir = r"D:\inpaint_result\CASIA_Distance"
    writer = SummaryWriter(os.path.join(save_dir, 'SR_Stage_4%s' % datetime.now().strftime("%Y%m%d-%H%M%S")))

    train_image_paths = r"C:\Users\_\Desktop\DB\CASIA_Ver4_Distance\reflection_trainset"
    train_mask_paths = r"D:/mask/CASIA_Distance/mask_trainset"
    train_gt_paths = r"C:\Users\8138\Desktop\DB\CASIA_Ver4_Distance\gt_trainset"
    train_large_mask_paths = r"D:\mask\CASIA_Distance\largemask_trainset"

    val_image_paths = r"C:\Users\_\Desktop\DB\CASIA_Ver4_Distance\reflection_validset"
    val_mask_paths = r"D:/mask/CASIA_Distance/mask_validset"
    val_gt_paths = r"C:\Users\8138\Desktop\DB\CASIA_Ver4_Distance\gt_validset"
    val_large_mask_paths = r"D:\mask\CASIA_Distance\largemask_validset"

    teacher_paths = r"D:\inpaint_result\CASIA_Distance\checkpoint_.tar"
    results_path = os.path.join(save_dir, "metrics.csv")

    MODEL_PATH = r"D:\recognition\CASIA_Distance\convnext_small_crop\saved_model_.pth"
    os.makedirs(save_dir, exist_ok=True)

    # checkpoint_path = "D:/inpaint_result/CASIA_Distance/checkpoint__.tar"
    checkpoint_path = None

    batch_size = 8
    lr = 0.0002
    num_epochs = 400
    lambda_adv = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    recognition_model = convnext_small(pretrained=False)
    recognition_model.head = nn.Identity()
    recognition_model.load_state_dict(torch.load(MODEL_PATH), strict=False)
    recognition_model = recognition_model.to(device)
    recognition_model.eval()

    train_dataset = InpaintDataset(train_image_paths, train_mask_paths, train_gt_paths, train_large_mask_paths)
    val_dataset = InpaintDataset(val_image_paths, val_mask_paths, val_gt_paths, val_large_mask_paths)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


    generator = Generator().to(device)
    generator_T = Generator_T().to(device)
    adaptor_enc2 = DAM(128, 256).to(device)
    adaptor_bottleneck = DAM(256,1024).to(device)
    adaptor_dec2 = DAM(128,256).to(device)

    checkpoint = torch.load(teacher_paths, map_location=device)
    generator_T.load_state_dict(checkpoint['generator_state_dict'])
    generator_T.eval()
    discriminator = Discriminator().to(device)

    optimizer_g = optim.Adam(list(generator.parameters()) +
    list(adaptor_enc2.parameters()) +
    list(adaptor_bottleneck.parameters()) +
    list(adaptor_dec2.parameters()), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))


    g_losses = []
    g_l2_losses = []
    g_adv_losses = []
    g_recog_losses = []
    d_losses = []
    kd_losses = []

    criterion = nn.BCELoss()


    with open(results_path, mode='w', newline='') as csvfile:
        writer_csv = csv.writer(csvfile)
        writer_csv.writerow([
            "epoch", "g_loss", "g_l2_loss", "g_adv_loss", "g_recog_loss", "d_loss", "kd_loss",
            "val_g_loss", "val_g_l2_loss", "val_g_adv_loss", "val_g_recog_loss", "val_d_loss", "val_kd_loss",
            "psnr", "ssim"
        ])


    start_epoch = 0

    if checkpoint_path:
        generator, discriminator, optimizer_g, optimizer_d, start_epoch, g_loss, g_l2_loss, g_adv_loss, d_loss \
            = load_checkpoint(checkpoint_path, generator, discriminator, optimizer_g, optimizer_d)


        print(f"Resuming training from epoch {start_epoch + 1}")


    for epoch in range(start_epoch, num_epochs):

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        g_loss, g_l2_loss, g_adv_loss, g_recog_loss, d_loss, kd_loss = train_gan_epoch(generator, generator_T, discriminator, adaptor_enc2, adaptor_bottleneck, adaptor_dec2
                                                                                       , train_loader, criterion, optimizer_g, optimizer_d, device, recognition_model, lambda_adv)
        g_losses.append(g_loss)
        g_l2_losses.append(g_l2_loss)
        g_adv_losses.append(g_adv_loss)
        g_recog_losses.append(g_recog_loss)
        d_losses.append(d_loss)
        kd_losses.append(kd_loss)

        val_g_loss, val_g_l2_loss, val_g_adv_loss, val_g_recog_loss, val_d_loss, val_kd_loss, psnr, ssim = validate_epoch(generator,generator_T,discriminator,
                            adaptor_enc2,adaptor_bottleneck,adaptor_dec2, val_loader, device, criterion, writer, recognition_model, lambda_adv, epoch, save_dir)


        writer.add_scalar("Train/G_Loss", g_loss, epoch)
        writer.add_scalar("Train/G_L2", g_l2_loss, epoch)
        writer.add_scalar("Train/G_adv", g_adv_loss, epoch)
        writer.add_scalar("Train/G_recog", g_recog_loss, epoch)
        writer.add_scalar("Train/D_Loss", d_loss, epoch)
        writer.add_scalar("Train/KD_Loss", kd_loss, epoch)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], G_Loss: {g_loss:.4f}, g_l2_Loss: {g_l2_loss:.4f}, g_adv_Loss: {g_adv_loss:.4f}, g_recog_Loss: {g_recog_loss:.4f} "
            f"d_Loss: {d_loss:.4f}, kd_Loss: {kd_loss:.4f}")
        print("----------validation-----------")
        print(
            f"val_g_loss: {val_g_loss:.4f}, val_g_l2_loss: {val_g_l2_loss:.4f} ,val_g_adv_loss: {val_g_adv_loss:.4f}, val_g_recog_loss: {val_g_recog_loss:.4f}, "
            f"val_d_loss: {val_d_loss:.4f}"
            f",val_kd_loss: {val_kd_loss:.4f} ,PSNR: {psnr:.4f}, SSIM: {ssim:.4f} ")

        with open(results_path, mode='a', newline='') as csvfile:
            writer_csv = csv.writer(csvfile)
            writer_csv.writerow([
                epoch + 1, g_loss, g_l2_loss, g_adv_loss, g_recog_loss, d_loss, kd_loss,
                val_g_loss, val_g_l2_loss, val_g_adv_loss, val_g_recog_loss, val_d_loss, val_kd_loss,
                psnr, ssim
            ])

        if epoch >= 100:
            torch.save({
                "epoch": epoch + 1,
                "generator_state_dict": generator.state_dict(),
                "g_loss": g_loss,
                "g_l2_loss": g_l2_loss,
                "g_adv_loss": g_adv_loss,
                "d_loss": d_loss,

                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict()

            }, os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.tar"))

    writer.close()


if __name__ == "__main__":
    main()
