import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

import itertools
import subprocess

from dataset import LoadDataset
from utils import set_grad, init_weights
from model import UNet, Discriminator

wandb.init(project="cycle-gan")


def run():
    SEED = 18
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    data_train = LoadDataset('data/edges2shoes/', 'train')  # cityscapes
    data_test = LoadDataset('data/edges2shoes/', 'val')  # cityscapes

    train_dataloader = DataLoader(data_train, batch_size=32, shuffle=True, num_workers=7)
    cycle_consistency = nn.L1Loss()
    discriminator_loss = nn.BCELoss()
    idt_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()

    lambda_idt = 0
    lambda_C = 0.5
    lambda_adv = 0
    beta1 = 0.5
    epochs = 5
    lr = 0.0003

    genAB = UNet(3, 3).cuda()
    init_weights(genAB, 'normal')
    genBA = UNet(3, 3).cuda()
    init_weights(genBA, 'normal')
    discrA = Discriminator(3).cuda()
    init_weights(discrA, 'normal')
    discrB = Discriminator(3).cuda()
    init_weights(discrB, 'normal')

    optG = torch.optim.AdamW(itertools.chain(
        genAB.parameters(), genBA.parameters()), lr=lr, betas=(beta1, 0.999))
    optD = torch.optim.AdamW(itertools.chain(
        discrA.parameters(), discrB.parameters()), lr=lr, betas=(beta1, 0.999))
    for epoch in range(epochs):
        for i, (batch_A, batch_B) in enumerate(train_dataloader):
            batch_A, batch_B = batch_A.cuda(), batch_B.cuda()
            optG.zero_grad()
            loss_G, loss_D = 0, 0
            fake_B = genAB(batch_A)
            cycle_A = genBA(fake_B)
            fake_A = genBA(batch_B)
            cycle_B = genAB(fake_A)
            if lambda_idt > 0:
                loss_G += idt_loss(fake_B, batch_B) * lambda_idt
                loss_G += idt_loss(fake_A, batch_A) * lambda_idt
            if lambda_C > 0:
                loss_G += cycle_consistency(cycle_A, batch_A) * lambda_C
                loss_G += cycle_consistency(cycle_B, batch_B) * lambda_C
            if lambda_adv > 0:
                set_grad([discrA, discrB], False)
                discr_feedbackA = discrA(fake_A)
                discr_feedbackB = discrB(fake_B)
                loss_G += discriminator_loss(discr_feedbackA,
                                             torch.ones_like(discr_feedbackA)) * lambda_adv
                loss_G += discriminator_loss(discr_feedbackB,
                                             torch.ones_like(discr_feedbackB)) * lambda_adv
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(itertools.chain(
                genAB.parameters(), genBA.parameters()), 10)
            optG.step()
            if lambda_adv > 0:
                set_grad([discrA, discrB], True)
                loss_D_fake, loss_D_true = 0, 0
                optD.zero_grad()
                logits = discrA(fake_A.detach())
                loss_D_fake += discriminator_loss(logits,
                                                  torch.zeros_like(logits))

                logits = discrB(fake_B.detach())
                loss_D_fake += discriminator_loss(logits,
                                                  torch.zeros_like(logits))
                loss_D_fake.backward()
                torch.nn.utils.clip_grad_norm_(itertools.chain(
                    discrA.parameters(), discrB.parameters()), 10)
                optD.step()

                optD.zero_grad()
                logits = discrA(batch_A)
                loss_D_true += discriminator_loss(logits,
                                                  torch.ones_like(logits))
                logits = discrB(batch_B)
                loss_D_true += discriminator_loss(logits,
                                                  torch.ones_like(logits))
                loss_D_true.backward()
                torch.nn.utils.clip_grad_norm_(itertools.chain(
                    discrA.parameters(), discrB.parameters()), 10)
                optD.step()
                loss_D = loss_D_fake + loss_D_true
                if (i % 200 == 0):
                    wandb.log({'train/loss_G': loss_G.item(), 'epoch': epoch},
                              len(train_dataloader) * epoch + i)
                    wandb.log({'train/pixel_error_A': l2_loss(fake_A, batch_A).mean().item(),
                               'epoch': epoch + 1}, len(train_dataloader) * epoch + i)
                    wandb.log({'train/pixel_error_B': l2_loss(fake_B, batch_B).mean().item(),
                               'epoch': epoch + 1}, len(train_dataloader) * epoch + i)
                    if lambda_adv:
                        wandb.log({'train/loss_D': loss_D.item()},
                                  len(train_dataloader) * epoch + i)
                        wandb.log({'train/mean_D_A': discr_feedbackA.mean().item()},
                                  len(train_dataloader) * epoch + i)
                        wandb.log({'train/mean_D_B': discr_feedbackB.mean().item()},
                                  len(train_dataloader) * epoch + i)
                    batch_i = 0
                    concat = torch.cat([fake_A[batch_i], batch_B[batch_i]], dim=-1)
                    wandb.log({f"fake_A_{str(epoch)}_{str(i)}": [wandb.Image((np.squeeze(concat.permute(
                        1, 2, 0).cpu().detach().numpy()) * 255).astype(np.int16), caption=str(i))], 'epoch': epoch + 1})
                    concat = torch.cat([fake_B[batch_i], batch_A[batch_i]], dim=-1)
                    wandb.log({f"fake_B_{str(epoch)}_{str(i)}": [wandb.Image((np.squeeze(concat.permute(
                        1, 2, 0).cpu().detach().numpy()) * 255).astype(np.int16), caption=str(i))], 'epoch': epoch + 1})


if __name__ == '__main__':
    subprocess.call("./download_data.sh", shell=True)
    run()
