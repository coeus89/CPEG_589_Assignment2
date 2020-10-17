import torch
from torch.autograd import Variable
import time as t
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
from utils.tensorboard_logger import Logger
from torchvision import utils
from models.WGAN_Clip import WGAN_Desc
from networks.DCGAN_MODEL import DCGAN_MODEL


# Note this is WGAN CLIP
class WGAN_CP(DCGAN_MODEL):
    def __init__(self, channels, generator_iters, batch_size):
        super().__init__(channels=channels, epochs=generator_iters, batch_size=batch_size)
        print("WGAN_CP init model.")
        self.D = WGAN_Desc(channels)
        self.G_filename = './WGAN_CLIP_generator.pkl'
        self.D_filename = './WGAN_CLIP_discriminator.pkl'
        self.img_filename = 'WGAN_CLIP_model_image.png'
        self.channels = channels
        self.generator_iters = generator_iters
        # check if cuda is available
        self.check_cuda()

        # WGAN values from paper
        self.learning_rate = 0.00005
        self.weight_cliping_limit = 0.01

        # WGAN with gradient clipping uses RMSprop instead of ADAM
        self.d_optimizer = torch.optim.RMSprop(self.D.parameters(), lr=self.learning_rate)
        self.g_optimizer = torch.optim.RMSprop(self.G.parameters(), lr=self.learning_rate)

        # Set the logger
        self.logger = Logger('./logs')
        self.logger.writer.flush()
        self.number_of_images = 10

        self.generator_iters = generator_iters
        self.critic_iter = 5


    # def check_cuda(self, cuda_flag=False):
    #     if cuda_flag:
    #         self.cuda_index = 0
    #         self.cuda = True
    #         self.D.cuda()
    #         self.G.cuda()
    #         print("Cuda enabled flag: {}".format(self.cuda))


    def train(self, train_loader):
        self.t_begin = t.time()
        #self.file = open("inception_score_graph.txt", "w")

        # Now batches are callable self.data.next()
        self.data = self.get_infinite_batches(train_loader)

        one = torch.FloatTensor([1])
        mone = one * -1
        if self.cuda:
            one = one.cuda()
            mone = mone.cuda()

        for g_iter in range(self.generator_iters):

            # Requires grad, Generator requires_grad = False
            for p in self.D.parameters():
                p.requires_grad = True

            # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                # Clamp parameters to a range [-c, c], c=self.weight_cliping_limit
                for p in self.D.parameters():
                    p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

                images = self.data.__next__()
                # Check for batch to have full batch_size
                if (images.size()[0] != self.batch_size):
                    continue

                z = torch.rand((self.batch_size, 100, 1, 1))

                if self.cuda:
                    images, z = Variable(images.cuda()), Variable(z.cuda())
                else:
                    images, z = Variable(images), Variable(z)


                # Train discriminator
                # WGAN - Training discriminator more iterations than generator
                # Train with real images
                d_loss_real = self.D(images)
                d_loss_real = d_loss_real.mean(0).view(1)
                d_loss_real.backward(one)

                # Train with fake images
                if self.cuda:
                    z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda()
                else:
                    z = Variable(torch.randn(self.batch_size, 100, 1, 1))
                fake_images = self.G(z)
                d_loss_fake = self.D(fake_images)
                d_loss_fake = d_loss_fake.mean(0).view(1)
                d_loss_fake.backward(mone)

                d_loss = d_loss_fake - d_loss_real
                Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()


            # Generator update
            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation

            self.G.zero_grad()

            # Train generator
            # Compute loss with fake images
            z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda()
            fake_images = self.G(z)
            g_loss = self.D(fake_images)
            g_loss = g_loss.mean().mean(0).view(1)
            g_loss.backward(one)
            g_cost = -g_loss
            self.g_optimizer.step()

            # Saving model and sampling images every 1000th generator iterations
            if (g_iter) % 1000 == 0:
                self.save_model()
                # Workaround because graphic card memory can't store more than 830 examples in memory for generating image
                # Therefore doing loop and generating 800 examples and stacking into list of samples to get 8000 generated images
                # This way Inception score is more correct since there are different generated examples from every class of Inception model
                # sample_list = []
                # for i in range(10):
                #     z = Variable(torch.randn(800, 100, 1, 1)).cuda(self.cuda_index)
                #     samples = self.G(z)
                #     sample_list.append(samples.data.cpu().numpy())
                #
                # # Flattening list of list into one list
                # new_sample_list = list(chain.from_iterable(sample_list))
                # print("Calculating Inception Score over 8k generated images")
                # # Feeding list of numpy arrays
                # inception_score = get_inception_score(new_sample_list, cuda=True, batch_size=32,
                #                                       resize=True, splits=10)

                if not os.path.exists('training_result_images/'):
                    os.makedirs('training_result_images/')

                # Denormalize images and save them in grid 8x8
                z = Variable(torch.randn(800, 100, 1, 1)).cuda(self.cuda_index)
                samples = self.G(z)
                samples = samples.mul(0.5).add(0.5)
                samples = samples.data.cpu()[:64]
                grid = utils.make_grid(samples)
                utils.save_image(grid, 'training_result_images/img_WGAN_generatori_iter_{}.png'.format(str(g_iter).zfill(3)))

                # Testing
                time = t.time() - self.t_begin
                #print("Inception score: {}".format(inception_score))
                print("Generator iter: {}".format(g_iter))
                print("Time {}".format(time))

                # Write to file inception_score, gen_iters, time
                #output = str(g_iter) + " " + str(time) + " " + str(inception_score[0]) + "\n"
                #self.file.write(output)


                # ============ TensorBoard logging ============#
                # (1) Log the scalar values

                if d_loss.shape.__len__() == 0:
                    discrim_loss = d_loss.item()
                else:
                    discrim_loss = d_loss.data[0]
                if g_loss.shape.__len__() == 0:
                    generator_loss = g_loss.item()
                else:
                    generator_loss = g_loss.data[0]
                if Wasserstein_D.shape.__len__() == 0:
                    Wasserstein_D_loss = Wasserstein_D.item()
                else:
                    Wasserstein_D_loss = Wasserstein_D.data[0]
                info = {
                    'Wasserstein distance': Wasserstein_D.data[0],
                    'Loss D': d_loss.data[0],
                    'Loss G': g_cost.data[0],
                    'Loss D Real': d_loss_real.data[0],
                    'Loss D Fake': d_loss_fake.data[0]

                }

                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, g_iter + 1)

                # (3) Log the images
                info = {
                    'real_images': self.real_images(images, self.number_of_images),
                    'generated_images': self.generate_img(z, self.number_of_images)
                }

                for tag, images in info.items():
                    self.logger.image_summary(tag, images, g_iter + 1)


        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))
        #self.file.close()

        # Save the trained parameters
        self.save_model()

    def real_images(self, images, number_of_images):
        if (self.C == 3):
            return self.to_np(images.view(-1, self.C, 32, 32)[:self.number_of_images])
        else:
            return self.to_np(images.view(-1, 32, 32)[:self.number_of_images])

    def get_infinite_batches(self, data_loader):
        while True:
            for i, (images, _) in enumerate(data_loader):
                yield images

    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        number_int = 10
        # interpolate between twe noise(z1, z2).
        z_intp = torch.FloatTensor(1, 100, 1, 1)
        z1 = torch.randn(1, 100, 1, 1)
        z2 = torch.randn(1, 100, 1, 1)
        if self.cuda:
            z_intp = z_intp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()

        z_intp = Variable(z_intp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            z_intp.data = z1*alpha + z2*(1.0 - alpha)
            alpha += alpha
            fake_im = self.G(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5) #denormalize
            images.append(fake_im.view(self.C,32,32).data.cpu())

        grid = utils.make_grid(images, nrow=number_int )
        utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved interpolated images.")