import torch
import torch.nn as nn
from models import GAN
from torch.autograd import Variable
# from torch.utils.tensorboard import SummaryWriter
import time as t
import os
from utils.tensorboard_logger import Logger
import torchvision.utils as tUtils
from networks.DCGAN_MODEL import DCGAN_MODEL

class GAN_MODEL(DCGAN_MODEL):
    def __init__(self, channels, epochs, batch_size):
        super().__init__(channels=channels, epochs=epochs, batch_size=batch_size)
        print("GAN model initialization.")
        self.G = GAN.GAN_Gen()
        self.D = GAN.GAN_Des()
        self.G_filename = './GAN_generator.pkl'
        self.D_filename = './GAN_discriminator.pkl'
        self.img_filename = 'gan_model_image.png'
        self.loss = nn.BCELoss()
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0002, weight_decay=0.00001)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=0.0002, weight_decay=0.00001)
        #self.logger = SummaryWriter(log_dir='./GANlogs', filename_suffix='.log')
        self.logger = Logger('./logs')
        self.number_of_images = 10
        self.check_cuda()


    def train(self, train_loader):
        self.t_begin = t.time()
        generator_iter = 0

        for epoch in range(self.epochs):
            self.epoch_start_time = t.time()

            for i, (images, _) in enumerate(train_loader):
            #for i, images in enumerate(train_loader):
                # this checks if there is a round number of batches
                if i == train_loader.dataset.__len__() // self.batch_size:
                    break

                # z = torch.randn((self.batch_size, 100, 1, 1))
                z = torch.randn((self.batch_size, 1, 1, 100))
                real_labels = torch.ones((self.batch_size, self.C))
                fake_labels = torch.zeros((self.batch_size, self.C))

                if self.cuda:
                    images, z = Variable(images).cuda(self.cuda_index), Variable(z).cuda(self.cuda_index)
                    real_labels = Variable(real_labels).cuda(self.cuda_index)
                    fake_labels = Variable(fake_labels).cuda(self.cuda_index)
                else:
                    images, z = Variable(images), Variable(z)
                    real_labels = Variable(real_labels)
                    fake_labels = Variable(fake_labels)

                # Train Discriminator
                # BCE loss is Binary Cross Entropy
                outputs = self.D(images)
                d_loss_real = self.loss(outputs, real_labels) # BCE Loss with real images

                # Fake Images
                if self.cuda:
                    z = Variable(torch.randn(self.batch_size, self.C, 1, 100)).cuda(self.cuda_index)
                else:
                    z = Variable(torch.randn(self.batch_size, self.C, 1, 100))

                fake_images = self.G(z)
                outputs = self.D(fake_images)
                d_loss_fake = self.loss(outputs, fake_labels)

                # Back Prop on Discriminator
                d_loss = d_loss_real + d_loss_fake
                self.D.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Train Generator
                # Compute loss with fake images

                if self.cuda:
                    z = Variable(torch.randn(self.batch_size, self.C, 1, 100)).cuda(self.cuda_index)
                else:
                    z = Variable(torch.randn(self.batch_size, self.C, 1, 100))

                fake_images = self.G(z)
                outputs = self.D(fake_images)
                g_loss = self.loss(outputs, real_labels)
                # g_loss = self.loss(outputs, fake_labels)

                # Back prop on Generator
                self.D.zero_grad()
                self.G.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()
                generator_iter += 1

                if ((i + 1) % 100) == 0:
                    if d_loss.shape.__len__() == 0:
                        discrim_loss = d_loss.item()
                    else:
                        discrim_loss = d_loss.data[0]
                    if g_loss.shape.__len__() == 0:
                        generator_loss = g_loss.item()
                    else:
                        generator_loss = g_loss.data[0]
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (i + 1), train_loader.dataset.__len__() // self.batch_size, discrim_loss, generator_loss))

                    # z = Variable(torch.randn(self.batch_size, 100).cuda(self.cuda_index))
                    z = Variable(torch.randn(self.batch_size, self.C, 100).to(self.device))

                    # Logging. Uses SummaryWriter from Tensorboard.
                    # Not sure if this is correct...
                    # (1) Log the scalar values
                    info = {
                        'd_loss': discrim_loss,
                        'g_loss': generator_loss
                    }

                    for tag, value in info.items():
                        self.logger.scalar_summary(tag, value, i + 1)

                    # (2) Log values and gradients of the parameters (histogram)
                    for tag, value in self.D.named_parameters():
                        tag = tag.replace('.', '/')
                        self.logger.histo_summary(tag, self.to_np(value), generator_iter)
                        self.logger.histo_summary(tag + '/grad', self.to_np(value.grad), generator_iter)

                    # (3) Log the images
                    info = {
                        'real_images': self.to_np(images.view(-1, 32, 32)[:self.number_of_images]),
                        'generated_images': self.generate_img(z, self.number_of_images)
                    }

                    # for tag, images in info.items():
                    #     self.logger.add_images(tag, images, i + 1)

                if generator_iter % 1000 == 0:
                    print('Generator iter-{}'.format(generator_iter))
                    self.save_model()

                    if not os.path.exists('training_result_images/'):
                        os.makedirs('training_result_images/')

                    # Denormalize images and save them in grid 8x8
                    z = Variable(torch.randn(self.batch_size, self.C, 100)).cuda(self.cuda_index)
                    samples = self.G(z)
                    samples = torch.reshape(samples, [samples.shape[0], self.C, 32, 32])
                    samples = samples.mul(0.5).add(0.5)
                    samples = samples.data.cpu()
                    grid = tUtils.make_grid(samples)
                    tUtils.save_image(grid, 'training_result_images/gan_image_iter_{}.png'.format(
                        str(generator_iter).zfill(3)))

        self.t_end = t.time()
        print('Time of training = {}'.format((self.t_end - self.t_begin)))
        # Save the trained parameters
        self.save_model()

    def evaluate(self):
        print(self.D_filename)
        print(self.G_filename)
        self.load_model()
        # self.load_model(self.D_filename, self.G_filename)
        # z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda(self.cuda_index)
        z = Variable(torch.randn(self.batch_size, self.C, 100).to(self.device))
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = tUtils.make_grid(samples)
        grid = torch.reshape(grid, [grid.shape[0], self.C, 32, 32])
        print("Grid of 8x8 images saved to {}].".format(self.img_filename))
        tUtils.save_image(grid, self.img_filename)