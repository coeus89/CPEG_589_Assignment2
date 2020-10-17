import torch
import torch.nn as nn
from models import GAN
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import time as t
import os
import torchvision.utils as tUtils
from networks.DCGAN_MODEL import DCGAN_MODEL

class GAN_MODEL(DCGAN_MODEL):
    def __init__(self, channels, epochs, batch_size):
        super().__init__(channels=channels, epochs=epochs, batch_size=batch_size)
        print("GAN model initialization.")
        self.G = GAN.GAN_Gen()
        self.D = GAN.GAN_Des()
        self.loss = nn.BCELoss()
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0002, weight_decay=0.00001)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=0.0002, weight_decay=0.00001)
        self.logger = SummaryWriter(log_dir='./GANlogs', filename_suffix='.log')
        self.number_of_images = 10


    def train(self, train_loader):
        self.t_begin = t.time()
        generator_iter = 0

        for epoch in range(self.epochs + 1):
            for i, (images, _) in enumerate(train_loader):
                # Check if round number of batches
                if i == train_loader.dataset.__len__() // self.batch_size:
                    break

                # Flatten image 1,32x32 to 1024
                images = images.view(self.batch_size, -1)
                z = torch.rand((self.batch_size, 100))

                real_labels = Variable(torch.ones(self.batch_size)).cuda(self.cuda_index)
                fake_labels = Variable(torch.zeros(self.batch_size)).cuda(self.cuda_index)

                if self.cuda:
                    images, z = Variable(images.cuda(self.cuda_index)), Variable(z.cuda(self.cuda_index))
                else:
                    images, z = Variable(images), Variable(z)

                # Train discriminator
                # compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
                # [Training discriminator = Maximizing discriminator being correct]
                outputs = self.D(images)
                d_loss_real = self.loss(outputs, real_labels)
                real_score = outputs

                # Compute BCELoss using fake images
                fake_images = self.G(z)
                outputs = self.D(fake_images)
                d_loss_fake = self.loss(outputs, fake_labels)
                fake_score = outputs

                # Optimize discriminator
                d_loss = d_loss_real + d_loss_fake
                self.D.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Train generator
                if self.cuda:
                    z = Variable(torch.randn(self.batch_size, 100).cuda(self.cuda_index))
                else:
                    z = Variable(torch.randn(self.batch_size, 100))
                fake_images = self.G(z)
                outputs = self.D(fake_images)

                # We train G to maximize log(D(G(z))[maximize likelihood of discriminator being wrong] instead of
                # minimizing log(1-D(G(z)))[minizing likelihood of discriminator being correct]
                # From paper  [https://arxiv.org/pdf/1406.2661.pdf]
                g_loss = self.loss(outputs, real_labels)

                # Optimize generator
                self.D.zero_grad()
                self.G.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()
                generator_iter += 1

                if ((i + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (i + 1), train_loader.dataset.__len__() // self.batch_size, d_loss.data[0],
                           g_loss.data[0]))

                    z = Variable(torch.randn(self.batch_size, 100).cuda(self.cuda_index))

                    # ============ TensorBoard logging ============#
                    # (1) Log the scalar values
                    info = {
                        'd_loss': d_loss.data[0],
                        'g_loss': g_loss.data[0]
                    }

                    for tag, value in info.items():
                        self.logger.scalar_summary(tag, value, i + 1)

                    # (2) Log values and gradients of the parameters (histogram)
                    for tag, value in self.D.named_parameters():
                        tag = tag.replace('.', '/')
                        self.logger.histo_summary(tag, self.to_np(value), i + 1)
                        self.logger.histo_summary(tag + '/grad', self.to_np(value.grad), i + 1)

                    # (3) Log the images
                    info = {
                        'real_images': self.to_np(images.view(-1, 32, 32)[:self.number_of_images]),
                        'generated_images': self.generate_img(z, self.number_of_images)
                    }

                    for tag, images in info.items():
                        self.logger.image_summary(tag, images, i + 1)

                if generator_iter % 1000 == 0:
                    print('Generator iter-{}'.format(generator_iter))
                    self.save_model()

                    if not os.path.exists('training_result_images/'):
                        os.makedirs('training_result_images/')

                    # Denormalize images and save them in grid 8x8
                    z = Variable(torch.randn(self.batch_size, 100)).cuda(self.cuda_index)
                    samples = self.G(z)
                    samples = samples.mul(0.5).add(0.5)
                    samples = samples.data.cpu()
                    grid = tUtils.make_grid(samples)
                    tUtils.save_image(grid, 'training_result_images/gan_image_iter_{}.png'.format(
                        str(generator_iter).zfill(3)))

        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))
        # Save the trained parameters
        self.save_model()

    def evaluate(self, D_model_filename, G_model_filename):
        self.load_model(D_model_filename, G_model_filename)
        z = Variable(torch.randn(self.batch_size, 100)).cuda(self.cuda_index)
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = tUtils.make_grid(samples)
        print("Grid of 8x8 images saved to 'gan_model_image.png'.")
        tUtils.save_image(grid, 'gan_model_image.png')

    def generate_img(self, z, number_of_images):
        samples = self.G(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            generated_images.append(sample.reshape(32, 32))
        return generated_images

    def save_model(self):
        torch.save(self.G.state_dict(), './GAN_generator.pkl')
        torch.save(self.D.state_dict(), './GAN_discriminator.pkl')

        print('Models save to ./GAN_generator.pkl & ./GAN_discriminator.pkl')

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))

    def to_np(self, x):
        return x.data.cpu().numpy()
