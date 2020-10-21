import torch
from utils.My_Enums import DatasetPicker as Dp
from utils.data_loader import DatasetLoader
from networks.GAN_MODEL import GAN_MODEL
from networks.DCGAN_MODEL import DCGAN_MODEL
from networks.WGAN_Model import WGAN_CP
from networks.WGAN_PENALTY_MODEL import WGAN_PENALTY


def main():

    # Normal GAN
    dataset_choice = Dp.CIFAR10
    # dataroot = 'datasets/mnist'
    dataroot = 'datasets'
    channels = 3
    epochs = 50
    batchsize = 50
    conv = False
    train_dataloader, test_dataloader = DatasetLoader.get_data_loader(dataset=dataset_choice, dataroot=dataroot,
                                                                      convolutional=conv, batch_size=batchsize)
    model = GAN_MODEL(channels=channels, epochs=epochs, batch_size=batchsize)
    # model.train(train_loader=train_dataloader)
    # model.evaluate(D_model_filename='GAN_discriminator.pkl', G_model_filename='GAN_generator.pkl')
    model.evaluate()


    # DC GAN
    dataset_choice = Dp.CIFAR10
    dataroot = 'datasets'
    channels = 3
    epochs = 45
    batchsize = 64
    conv = True
    train_dataloader, test_dataloader = DatasetLoader.get_data_loader(dataset=dataset_choice, dataroot=dataroot,
                                                                      convolutional=conv, batch_size=batchsize)
    model = DCGAN_MODEL(channels=channels, epochs=epochs, batch_size=batchsize)
    model.train(train_loader=train_dataloader)
    model.evaluate()


    # WGAN clip
    dataset_choice = Dp.CIFAR10
    dataroot = 'datasets'
    channels = 3
    generator_iters = 60000
    batchsize = 64
    conv = True
    train_dataloader, test_dataloader = DatasetLoader.get_data_loader(dataset=dataset_choice, dataroot=dataroot,
                                                                      convolutional=conv, batch_size=batchsize)
    model = WGAN_CP(channels=channels, generator_iters=generator_iters, batch_size=batchsize)
    model.train(train_loader=train_dataloader)
    model.evaluate()


    # WGAN Penalty
    dataset_choice = Dp.CIFAR10
    dataroot = 'datasets'
    channels = 3
    generator_iters = 60000
    batchsize = 64
    conv = True
    train_dataloader, test_dataloader = DatasetLoader.get_data_loader(dataset=dataset_choice, dataroot=dataroot,
                                                                      convolutional=conv, batch_size=batchsize)
    model = WGAN_PENALTY(channels=channels, generator_iters=generator_iters, batch_size=batchsize)
    model.train(train_loader=train_dataloader)
    model.evaluate()



if __name__ == '__main__':
    main()
