"""
code made by Korneliusz Lewczuk korneliuszlewczuk@gmail.com
sorces:
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
"""

import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from dataloaders import DataloaderGenerator, DataloaderDiscriminator
from models import Generator, Discriminator
from utils import denormalize_tensor
import time

epochs_num = 1000
batch_size = 150
lr = 0.00002
sample_interval = 15000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.makedirs("images", exist_ok=True)

criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss() # weight == 1
criterion_identity = torch.nn.L1Loss() # weight == 0.1

generator = Generator()
discriminator = Discriminator()

# load state if u can discriminator_ver1
# generator.load_state_dict(torch.load())
# discriminator.load_state_dict(torch.load())
generator.to(device)
discriminator.to(device)
generator.train()
discriminator.train()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)
scaler_G = torch.cuda.amp.GradScaler()
scaler_D = torch.cuda.amp.GradScaler()

dataset_path = ''
dataloader_generator = torch.utils.data.DataLoader(DataloaderGenerator(dataset_path),
                                                    batch_size=batch_size, shuffle=True, drop_last=True)
dataloader_discriminator = torch.utils.data.DataLoader(DataloaderDiscriminator(dataset_path),
                                                        batch_size=batch_size, shuffle=True, drop_last=True)

torch.autograd.set_detect_anomaly(True)

for epoch in range(epochs_num):
    start_epoch_time = time.time()
    dataloader_generator_lenght =  len(dataloader_generator)
    print("len(dataloader): ", dataloader_generator_lenght)
    for i, (cloth, person_with_cloth, person_with_cloth_mask, target_cloth) in enumerate(dataloader_generator):

        real_cloth = cloth.to(device)
        real_person_with_cloth = person_with_cloth.to(device)
        real_target_cloth = target_cloth.to(device)
        person_with_cloth_mask = person_with_cloth_mask.to(device)

        # Adversarial ground truths
        valid = Variable(torch.FloatTensor(batch_size, 1).fill_(0.9), requires_grad=False).to(device)
        fake = Variable(torch.FloatTensor(batch_size, 1).fill_(0.1), requires_grad=False).to(device)

        # -----------------
        #  Train Generators
        # -----------------

        optimizer_G.zero_grad()

        with torch.cuda.amp.autocast():
            # 1 cykl cycle-gana
            input_to_generator1 = torch.cat((real_cloth, real_person_with_cloth, real_target_cloth), dim=1)
            output_from_generator1 = generator(input_to_generator1)
            generated_img1, mask1 = torch.split(output_from_generator1, [3, 1], dim=1)
            # identyty loss, czyli czy zdjecia sa identyczne po tym jak przjda przez nn
            loss_identity_mask1 =  criterion_identity(mask1, person_with_cloth_mask)
            # Gan
            input_to_discriminator_gen_train1 = torch.cat((generated_img1, real_target_cloth), dim=1)
            output_form_discriminator_gen_train1 = discriminator(input_to_discriminator_gen_train1)
            # GAN loss
            loss_gan1 = criterion_GAN(output_form_discriminator_gen_train1, valid)
            # 2 cykl cycle-gana
            input_to_generator2 = torch.cat((real_target_cloth, generated_img1, real_cloth), dim=1)
            output_from_generator2 = generator(input_to_generator2)
            generated_img2, mask2 = torch.split(output_from_generator2, [3, 1], dim=1)
            # identyty loss
            loss_identity_mask2 =  criterion_identity(mask2, person_with_cloth_mask)
            # Gan
            input_to_discriminator_gen_train2 = torch.cat((generated_img2, real_cloth), dim=1)
            output_form_discriminator_gen_train2 = discriminator(input_to_discriminator_gen_train2)
            # GAN loss
            loss_gan2 = criterion_GAN(output_form_discriminator_gen_train2, valid)
            # cycle
            loss_cycle_gen = criterion_cycle(generated_img2, real_person_with_cloth)
            # loses
            loss_identity_gen = (loss_identity_mask1 + loss_identity_mask2) / 2
            loss_gan = (loss_gan1+loss_gan2)/2
            g_loss = loss_gan + 0.1 * loss_identity_gen + loss_cycle_gen

        scaler_G.scale(g_loss).backward()
        scaler_G.step(optimizer_G)
        scaler_G.update()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        if i % 5 == 0:
            optimizer_D.zero_grad()

            generated_img1 = generated_img1.detach()
            generated_img2 = generated_img2.detach()

            with torch.cuda.amp.autocast():
                # Loss for real images 1
                cloth, person_with_cloth = next(iter(dataloader_discriminator))
                real_cloth = cloth.to(device)
                real_person_with_cloth = person_with_cloth.to(device)

                input_to_discriminator_dis_train_real = torch.cat((real_person_with_cloth, real_cloth), dim=1)
                output_form_discriminator_dis_train = discriminator(input_to_discriminator_dis_train_real)

                d_real_loss = criterion_GAN(output_form_discriminator_dis_train, valid)
                # Loss for real images 2
                cloth, person_with_cloth = next(iter(dataloader_discriminator))
                real_cloth = cloth.to(device)
                real_person_with_cloth = person_with_cloth.to(device)

                input_to_discriminator_dis_train_real = torch.cat((real_person_with_cloth, real_cloth), dim=1)
                output_form_discriminator_dis_train = discriminator(input_to_discriminator_dis_train_real)

                d_real_loss = criterion_GAN(output_form_discriminator_dis_train, valid)
                # Loss for fake images 1
                input_to_discriminator_dis_train_fake1 = torch.cat((generated_img1, real_target_cloth), dim=1)

                output_form_discriminator_dis_train1 = discriminator(input_to_discriminator_dis_train_fake1)

                d_fake_loss1 = criterion_GAN(output_form_discriminator_dis_train1, fake)
                # Loss for fake images 2
                input_to_discriminator_dis_train_fake2 = torch.cat((generated_img2, real_cloth), dim=1)

                output_form_discriminator_dis_train2 = discriminator(input_to_discriminator_dis_train_fake2)

                d_fake_loss = criterion_GAN(output_form_discriminator_dis_train2, fake)
                d_loss = (d_real_loss + d_fake_loss) / 2

            scaler_D.scale(d_loss).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()


        print(
            "[Epoch ", epoch,"] [D loss: ,",d_loss.item(),",] [G loss: ",g_loss.item(),"]"
        )
        images_done =  ((epoch*dataloader_generator_lenght)+(i+1))*batch_size
        generator_model_name = "generator_ver1.0_epoch_"+str(epoch)+"_images_done_"+str(images_done)+"_.pth"
        generator_model_path = os.path.join("", generator_model_name)

        discriminator_model_name = "discriminator_ver1.0_epoch_"+str(epoch)+"_images_done_"+str(images_done)+"_.pth"
        discriminator_model_path = os.path.join("", discriminator_model_name)

        if images_done % sample_interval == 0:
            save_image([tensor for idx, tensor in enumerate(generated_img1.data) if idx <16], "images_"+str(images_done)+".png", normalize=True, nrow=4)
            save_image([tensor for idx, tensor in enumerate(mask1.data) if idx <16], "masks_"+str(images_done)+".png", normalize=True, nrow=4)
            torch.save(generator.state_dict(), generator_model_path)
            torch.save(discriminator.state_dict(), discriminator_model_path)
    time_per_epoch = time.time() - start_epoch_time
    print("time per epoch: ",time_per_epoch , " przewidowany czas: ", time_per_epoch*epochs_num)
