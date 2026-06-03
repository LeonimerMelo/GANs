# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 16:47:36 2025

@author: Leonimer

References
==========
https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf
https://arxiv.org/pdf/1511.06434.pdf

Introduction
============
This tutorial will give an introduction to DCGANs through an example. We will train a 
generative adversarial network (GAN) to generate new celebrities after showing it 
pictures of many real celebrities. Most of the code here is from the DCGAN implementation 
in pytorch/examples, and this document will give a thorough explanation of the 
implementation and shed light on how and why this model works. But don’t worry, no 
prior knowledge of GANs is required, but it may require a first-timer to spend some 
time reasoning about what is actually happening under the hood. Also, for the sake 
of time it will help to have a GPU, or two. Lets start from the beginning.
""" 
  
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt


# set for training, reset for inference
isTraining = True
isTraining = False

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

# Root directory for dataset
#dataroot = "data/celeba"
dataroot = 'C:\\Leo\\python scripts\\data\\celeba'
# Number of workers for dataloader
workers = 2
# Batch size during training
batch_size = 128
# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Learning rate for optimizers
lr = 0.0001
# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

'''
Dataset
=======
In this tutorial we will use the Celeb-A Faces dataset which can be downloaded at 
the linked site, or in Google Drive. The dataset will download as a file named 
img_align_celeba.zip. Once downloaded, create a directory named celeba and extract 
the zip file into that directory. Then, set the dataroot input for this notebook 
to the celeba directory you just created. 
This is an important step because we will be using the ImageFolder dataset class, 
which requires there to be subdirectories in the dataset root folder. Now, we can 
create the dataset, create the dataloader, set the device to run on, and finally 
visualize some of the training data.
'''
# We can use an image folder dataset the way we have it setup. Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
# Dataset e batchs sizes
print('dataset number of samples:', len(dataloader.dataset))
print('batch size:', batch_size)
print('number of batches:', len(dataloader))

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

'''
Weight Initialization
=====================
From the DCGAN paper, the authors specify that all model weights shall be randomly 
initialized from a Normal distribution with mean=0, stdev=0.02. The weights_init 
function takes an initialized model as input and reinitializes all convolutional, 
convolutional-transpose, and batch normalization layers to meet this criteria. 
This function is applied to the models immediately after initialization.
'''
# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
'''
Generator
=========
The generator, G, is designed to map the latent space vector (z) to data-space. 
Since our data are images, converting z to data-space means ultimately creating a 
RGB image with the same size as the training images (i.e. 3x64x64). In practice, 
this is accomplished through a series of strided two dimensional convolutional 
transpose layers, each paired with a 2d batch norm layer and a relu activation. 
The output of the generator is fed through a tanh function to return it to the input 
data range of [−1,1]. It is worth noting the existence of the batch norm functions 
after the conv-transpose layers, as this is a critical contribution of the DCGAN paper. 
These layers help with the flow of gradients during training.
'''
# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)

'''
Now, we can instantiate the generator and apply the weights_init function. Check 
out the printed model to see how the generator object is structured.
'''
# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights
#  to ``mean=0``, ``stdev=0.02``.
netG.apply(weights_init)

# Print the model
print(netG)

'''
Discriminator
As mentioned, the discriminator, D, is a binary classification network that takes 
an image as input and outputs a scalar probability that the input image is real 
(as opposed to fake). Here, D takes a 3x64x64 input image, processes it through a 
series of Conv2d, BatchNorm2d, and LeakyReLU layers, and outputs the final probability 
through a Sigmoid activation function. This architecture can be extended with more 
layers if necessary for the problem, but there is significance to the use of the 
strided convolution, BatchNorm, and LeakyReLUs. The DCGAN paper mentions it is a 
good practice to use strided convolution rather than pooling to downsample because 
it lets the network learn its own pooling function. Also batch norm and leaky relu 
functions promote healthy gradient flow which is critical for the learning process 
of both G and D.
'''
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    
# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights
# like this: ``to mean=0, stdev=0.2``.
netD.apply(weights_init)

# Print the model
print(netD)

'''
Loss Functions and Optimizers
=============================
With D and G setup, we can specify how they learn through the loss functions and 
optimizers. We will use the Binary Cross Entropy loss (BCELoss) function which is 
defined in PyTorch as:
    
ℓ(x,y)=L={l1,…,ln}⊤, 
ln=−[yn⋅log(xn)+(1−yn)⋅log(1−xn)]

Notice how this function provides the calculation of both log components in the 
objective function (i.e. log(D(x)) and log(1−D(G(z)))). We can specify what part 
of the BCE equation to use with the y input. This is accomplished in the training 
loop which is coming up soon, but it is important to understand how we can choose 
which component we wish to calculate just by changing y (i.e. GT labels).
Next, we define our real label as 1 and the fake label as 0. These labels will be 
used when calculating the losses of D and G, and this is also the convention used 
in the original GAN paper. Finally, we set up two separate optimizers, one for D and 
one for G. As specified in the DCGAN paper, both are Adam optimizers with learning 
rate 0.0002 and Beta1 = 0.5. For keeping track of the generator’s learning progression, 
we will generate a fixed batch of latent vectors that are drawn from a Gaussian 
distribution (i.e. fixed_noise) . In the training loop, we will periodically input 
this fixed_noise into G, and over the iterations we will see images form out of 
the noise.
'''
# Initialize the Binary Cross Entropy loss (BCELoss) function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

'''
Training
========
Finally, now that we have all of the parts of the GAN framework defined, we can 
train it. Be mindful that training GANs is somewhat of an art form, as incorrect 
hyperparameter settings lead to mode collapse with little explanation of what went 
wrong. Here, we will closely follow Algorithm 1 from the Goodfellow’s paper, while 
abiding by some of the best practices shown in ganhacks. Namely, we will “construct 
different mini-batches for real and fake” images, and also adjust G’s objective function 
to maximize log(D(G(z))). Training is split up into two main parts. Part 1 updates 
the Discriminator and Part 2 updates the Generator

Part 1 - Train the Discriminator
================================
Recall, the goal of training the discriminator is to maximize the probability of 
correctly classifying a given input as real or fake. In terms of Goodfellow, we 
wish to “update the discriminator by ascending its stochastic gradient”. Practically, 
we want to maximize log(D(x))+log(1−D(G(z))). Due to the separate mini-batch suggestion 
from ganhacks, we will calculate this in two steps. First, we will construct a batch 
of real samples from the training set, forward pass through D, calculate the 
loss(log(D(x))), then calculate the gradients in a backward pass. Secondly, we will 
construct a batch of fake samples with the current generator, forward pass this batch 
through D, calculate the loss(log(1−D(G(z)))), and accumulate the gradients with a 
backward pass. Now, with the gradients accumulated from both the all-real and all-fake 
batches, we call a step of the Discriminator’s optimizer.

Part 2 - Train the Generator
============================
As stated in the original paper, we want to train the Generator by minimizing 
log(1−D(G(z))) in an effort to generate better fakes. As mentioned, this was shown 
by Goodfellow to not provide sufficient gradients, especially early in the learning 
process. As a fix, we instead wish to maximize log(D(G(z))). In the code we accomplish 
this by: classifying the Generator output from Part 1 with the Discriminator, computing 
G’s loss using real labels as GT, computing G’s gradients in a backward pass, and 
finally updating G’s parameters with an optimizer step. It may seem counter-intuitive 
to use the real labels as GT labels for the loss function, but this allows us to use 
the log(x) part of the BCELoss (rather than the log(1−x) part) which is exactly what 
we want.
'''
# Training Loop
G_losses = []
D_losses = []
iters = 0
# Number of training epochs
num_epochs = 30
if isTraining:
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
    
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()
    
            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()
    
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
    
            # Output training stats and fake images
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch+1, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_last = vutils.make_grid(fake, padding=2, normalize=True)
                
                plt.figure(figsize=(8,8))
                plt.axis("off")
                plt.title("Fake Images - epoch #" + str(epoch+1)+' - iter: #'+str(iters))
                plt.imshow(np.transpose(vutils.make_grid(fake[:64], padding=2, normalize=True).cpu(),(1,2,0)))
                plt.show()
    
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
        
            iters += 1
  
    '''
    Results
    =======
    Finally, lets check out how we did. Here, we will look at three different results. 
    First, we will see how D and G’s losses changed during training. Second, we will 
    visualize G’s output on the fixed_noise batch for every epoch. And third, we will 
    look at a batch of real data next to a batch of fake data from G.
    '''
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))
    # Plot the real images
    plt.figure(figsize=(15,15))
    # plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
    # plt.show()
    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_last,(1,2,0)))
    plt.show()

# path for saving and load GAN trained model
path = 'C:\\Leo\\python scripts\\GANs\\'

if isTraining:
    '''
    Saving Model for inference
    ==========================
    A common way to save a model is to serialize the internal state dictionary 
    (containing the model parameters).
    '''
    torch.save(netG.state_dict(), path + "GAN_celeba_"+str(num_epochs)+".pth")
    print("Saved PyTorch Model State to GAN_celeba_"+str(num_epochs)+".pth")

if not isTraining:
    # Loading Model for inference
    netG = Generator(nz).to(device)
    netG.load_state_dict(torch.load(path + "GAN_celeba_"+str(num_epochs)+".pth", weights_only=True))
    print("Loaded PyTorch Model State from GAN_celeba_"+str(num_epochs)+".pth")
    
    manualSeed = random.randint(1, 1000) # use if you want new results
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    # plot generated images with pretreined GAN               
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()

    plt.figure(figsize=(8,8))
    plt.title("Fake Images - training epochs: "+str(num_epochs))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
    
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))
    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
    # Plot the fake images
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images: "+str(num_epochs)+" epochs")
    plt.imshow(np.transpose(vutils.make_grid(fake, normalize=True).cpu(),(1,2,0)))
    plt.show()
    
    # plt.figure(figsize=(1,1))
    # plt.axis("off")
    plt.title("One fake image", fontsize=10)
    plt.imshow(np.transpose(vutils.make_grid(fake[0], normalize=True).cpu(),(1,2,0)))
    plt.show()