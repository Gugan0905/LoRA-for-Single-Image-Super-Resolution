"CS7180 Advanced Perception  12/13/2023   Anirudh Muthuswamy, Gugan Kathiresan, Aditya Varshney"


import torch
import time
import os
import numpy as np
from torch import nn
from PIL import Image
from model.model import RealESRGAN
from loadGenerator import load_generator
import LoRAParametrization
from torchvision.utils import save_image
from model.Discriminator import UNetDiscriminatorSN
from model.FeatureExtractor import FeatureExtractor
from DatasetFromFolder import TrainDatasetFromFolder
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from EMA import EMA

#helper method to calculate psnr with hr and sr images

def calculate_psnr(hr_image, sr_image):
    hr_array = np.array(hr_image).astype(np.uint8)
    sr_array = np.array(sr_image).astype(np.uint8)
    return peak_signal_noise_ratio(hr_array, sr_array)

#helper method to calculate ssim with hr and sr images

def calculate_ssim(hr_image, sr_image):
    hr_array = np.array(hr_image)
    sr_array = np.array(sr_image)
    return structural_similarity(hr_array, sr_array, multichannel=True)

device = torch.device('cuda')

#The class below is used to setup the generator, discriminator, optimizers, loss and ema hyperparameters
# for finetuning using lora parameterization. 
#It contains methods to train and get inference times for the model as well.

class SetupESRGANLoRA():

    def __init__(self, generator_weight_path, discriminator_weight_path, learning_rate):
        
        self.generator = load_generator(weight_path = generator_weight_path)
        LoRAParametrization.add_lora(self.generator)

        for name, param in self.generator.named_parameters():
            if 'lora' not in name:
                # print(f'Freezing non-LoRA parameter {name}')
                param.requires_grad = False

        self.discriminator = UNetDiscriminatorSN(3).to(device)
        self.discriminator.load_state_dict(torch.load(discriminator_weight_path))
        self.feature_extractor = FeatureExtractor().to(device)


        # set feature extractor to inference mode
        self.feature_extractor.eval()

        # Losses
        self.criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
        self.criterion_content = torch.nn.L1Loss().to(device)
        self.criterion_pixel = torch.nn.L1Loss().to(device)
        self.lora_state_dict = None

        # initialize optimzier

        self.parameters = [
            {"params": list(LoRAParametrization.get_lora_params(self.generator))},
        ]

        self.generator.to(device)

        lr = learning_rate

        self.optimizer_G = torch.optim.Adam(self.parameters, lr=lr)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        # initialize ema
        self.ema_G = EMA(self.generator, 0.999)
        self.ema_D = EMA(self.discriminator, 0.999)
        self.ema_G.register()
        self.ema_D.register()

    #method used to train the model given input training parameters.

    def train(self,dataset_path, crop_size, batch_size, warmup_batches, 
          n_batches, batch, sample_interval,save_path_suffix,lora_save_path, device = 'cuda'):

        train_set = TrainDatasetFromFolder(dataset_path, crop_size=crop_size, upscale_factor = 4)
        train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=batch_size, shuffle=True)
        
        batch = 0
        total_psnr = []
        total_ssim = []


        while batch < n_batches:
            for i, (data, target) in enumerate(train_loader):
                batches_done = batch + i

                imgs_lr = data.to(device)
                imgs_hr = target.to(device)

                valid = torch.ones((imgs_lr.size(0), 1, *imgs_hr.shape[-2:]), requires_grad=False).to(device)
                fake = torch.zeros((imgs_lr.size(0), 1, *imgs_hr.shape[-2:]), requires_grad=False).to(device)

                # ---------------------
                # Training Generator
                # ---------------------

                self.optimizer_G.zero_grad()

                gen_hr = self.generator(imgs_lr)

                # Compute PSNR and SSIM
                with torch.no_grad():
                    gen_hr_np = gen_hr.detach().cpu().numpy().transpose(0, 2, 3, 1)  # Convert tensor to NumPy (NCHW to NHWC)
                    imgs_hr_np = imgs_hr.cpu().numpy().transpose(0, 2, 3, 1)  # Convert tensor to NumPy (NCHW to NHWC)

                    psnr_values = []
                    ssim_values = []

                    for gen_img, hr_img in zip(gen_hr_np, imgs_hr_np):
                        # Ensure pixel values are in the range [0, 1]
                        gen_img = np.clip(gen_img, 0.0, 1.0)
                        hr_img = np.clip(hr_img, 0.0, 1.0)

                        psnr = compare_psnr(hr_img, gen_img)
                        ssim = compare_ssim(hr_img, gen_img, multichannel=True)

                        psnr_values.append(psnr)
                        ssim_values.append(ssim)

                    avg_psnr = np.mean(psnr_values)
                    avg_ssim = np.mean(ssim_values)

                    total_psnr.append(avg_psnr)
                    total_ssim.append(avg_ssim)

                    # Print or log the average PSNR and SSIM for this batch
                    print('[Iteration %d/%d] [Batch %d/%d] [PSNR: %.4f] [SSIM: %.4f]' % (batches_done, n_batches, i, len(train_loader), avg_psnr, avg_ssim))

                loss_pixel = self.criterion_pixel(gen_hr, imgs_hr)

                if batches_done < warmup_batches:
                    loss_pixel.backward()
                    self.optimizer_G.step()
                    self.ema_G.update()
                    print(
                        '[Iteration %d/%d] [Batch %d/%d] [G pixel: %f]' %
                        (batches_done, n_batches, i, len(train_loader), loss_pixel.item())
                    )
                    continue
                elif batches_done == warmup_batches:
                    self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=1e-4)

                pred_real = self.discriminator(imgs_hr).detach()
                pred_fake = self.discriminator(gen_hr)

                loss_GAN = (
                    self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid) +
                    self.criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), fake)
                ) / 2

                gen_features = self.feature_extractor(gen_hr)
                real_features = self.feature_extractor(imgs_hr)
                real_features = [real_f.detach() for real_f in real_features]
                loss_content = sum(self.criterion_content(gen_f, real_f) * w for gen_f, real_f, w in zip(gen_features, real_features, [0.1, 0.1, 1, 1, 1]))

                loss_G = loss_content + 0.1 * loss_GAN + loss_pixel

                loss_G.backward()
                self.optimizer_G.step()
                self.ema_G.update()

                # ---------------------
                # Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                pred_real = self.discriminator(imgs_hr)
                pred_fake = self.discriminator(gen_hr.detach())

                loss_real = self.criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
                loss_fake = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

                loss_D = (loss_real + loss_fake) / 2

                loss_D.backward()
                self.optimizer_D.step()
                self.ema_D.update()

                # -------------------------
                # Log Progress
                # -------------------------

                print(
                    '[Iteration %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]' %
                    (
                        batches_done,
                        n_batches,
                        i,
                        len(train_loader),
                        loss_D.item(),
                        loss_G.item(),
                        loss_content.item(),
                        loss_GAN.item(),
                        loss_pixel.item()
                    )
                )

                if batches_done % sample_interval == 0:
                    imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4, mode='bicubic')
                    img_grid = torch.clamp(torch.cat((imgs_lr, gen_hr, imgs_hr), -1), min=0, max=1)
                    save_image(img_grid, f'images/training/{save_path_suffix}_%d.png' % batches_done, nrow=1, normalize=False)

            batch = batches_done + 1

            self.ema_G.apply_shadow()
            self.ema_D.apply_shadow()

            torch.save(self.generator.state_dict(), f'saved_models/generator_{save_path_suffix}_%d.pth' % batch)
            torch.save(self.discriminator.state_dict(), f'saved_models/discriminator_{save_path_suffix}_%d.pth' % batch)

            self.ema_G.restore()
            self.ema_D.restore()

        print("total_psnr: ", np.mean(total_psnr))
        print("total_ssim: ", np.mean(total_ssim))

        self.lora_state_dict = LoRAParametrization.get_lora_state_dict(self.generator)

        torch.save(self.lora_state_dict, lora_save_path)

    #Method used to get the overall inference time given the generator pretrained paths 
    # and high resolution dataset path

    def get_inference_time(self,generator_weight_path, hr_dataset_path):
    
        generator = load_generator(weight_path = generator_weight_path )
    
        if self.lora_state_dict != None:
        
            LoRAParametrization.add_lora(generator)
            _ = generator.load_state_dict(self.lora_state_dict, strict=False)
            LoRAParametrization.merge_lora(generator)
        
        generator.to(device)
        generator.eval()

        model = RealESRGAN(device, generator, scale=4)

        image_files = [f for f in os.listdir(hr_dataset_path) if os.path.isfile(os.path.join(hr_dataset_path, f))]

        total_time = 0
    
        total_psnr = []
        total_ssim = []

        # Loop through each image in the dataset
        for num, image_file in enumerate(image_files):
            path_to_image = os.path.join(hr_dataset_path, image_file)
            hr_image = Image.open(path_to_image).convert('RGB')
            lr_image = hr_image.resize((hr_image.size[0]//4, hr_image.size[1]//4))
    
            start_time = time.time()
            sr_image = model.predict(lr_image)
            end_time = time.time()
            inference_time = end_time - start_time
            total_time += inference_time
        
            psnr = calculate_psnr(hr_image, sr_image)
            ssim = calculate_ssim(hr_image, sr_image)

            total_psnr.append(psnr)
            total_ssim.append(ssim)

            print(f"Inference time for image {num}: {inference_time} seconds")

        return total_time / len(image_files), np.mean(total_psnr), np.mean(total_ssim)

#Main method to carry out fine-tuning using lora for ESRGAN 

if __name__ == '__main__':

    generator_weight_path = "/content/drive/MyDrive/cs7180-Final-Project/RealESRGAN_x4.pth"
    discriminator_weight_path  = "/content/drive/MyDrive/cs7180-Final-Project/discriminator_2025.pth"
    learning_rate = 0.0002

    esr = SetupESRGANLoRA(generator_weight_path, discriminator_weight_path, learning_rate)

    dataset_path='/content/drive/MyDrive/cs7180-Final-Project/FloodNet-Supervised_v1.0/train/train-org-img'
    crop_size = 64
    upscale_factor = 4
    batch_size = 4
    warmup_batches = 100
    n_batches = 500
    residual_blocks = 23
    batch = 0
    lr = 0.0002
    sample_interval = 5
    channels = 3

    save_path_suffix = 'lora_set14_96inp'
    lora_save_path = './lora_weights/500Iter_esrgan_384hr_set14_finetune.pth'

    esr.train(dataset_path, crop_size, batch_size, warmup_batches, n_batches, batch, sample_interval, save_path_suffix,lora_save_path, device = 'cuda')

    esr.get_inference_time("/content/drive/MyDrive/cs7180-Final-Project/RealESRGAN_x4.pth",'/content/drive/MyDrive/cs7180-Final-Project/FloodNet-Supervised_v1.0/train/train-org-img')






    












