#!/usr/bin/env python
# coding: utf-8

# # Face Generation
# 
# In this project, you'll define and train a Generative Adverserial network of your own creation on a dataset of faces. Your goal is to get a generator network to generate *new* images of faces that look as realistic as possible!
# 
# The project will be broken down into a series of tasks from **defining new architectures training adversarial networks**. At the end of the notebook, you'll be able to visualize the results of your trained Generator to see how it performs; your generated samples should look like fairly realistic faces with small amounts of noise.
# 
# ### Get the Data
# 
# You'll be using the [CelebFaces Attributes Dataset (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to train your adversarial networks.
# 
# This dataset has higher resolution images than datasets you have previously worked with (like MNIST or SVHN) you've been working with, and so, you should prepare to define deeper networks and train them for a longer time to get good results. It is suggested that you utilize a GPU for training.
# 
# ### Pre-processed Data
# 
# Since the project's main focus is on building the GANs, we've done *some* of the pre-processing for you. Each of the CelebA images has been cropped to remove parts of the image that don't include a face, then resized down to 64x64x3 NumPy images. Some sample data is show below.
# 
# <img src='assets/processed_face_data.png' width=60% />
# 
# > If you are working locally, you can download this data [by clicking here](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5be7eb6f_processed-celeba-small/processed-celeba-small.zip)
# 
# This is a zip file that you'll need to extract in the home directory of this notebook for further loading and processing. After extracting the data, you should be left with a directory of data `processed-celeba-small/`.

# In[1]:


# run this once to unzip the file
get_ipython().system('unzip processed-celeba-small.zip')


# In[1]:


from glob import glob
from typing import Tuple, Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transform

import tests


# In[2]:


data_dir = 'processed_celeba_small/celeba/'


# ## Data pipeline
# 
# The [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset contains over 200,000 celebrity images with annotations. Since you're going to be generating faces, you won't need the annotations, you'll only need the images. Note that these are color images with [3 color channels (RGB)](https://en.wikipedia.org/wiki/Channel_(digital_image)#RGB_Images) each.
# 
# ### Pre-process and Load the Data
# 
# Since the project's main focus is on building the GANs, we've done *some* of the pre-processing for you. Each of the CelebA images has been cropped to remove parts of the image that don't include a face, then resized down to 64x64x3 NumPy images. This *pre-processed* dataset is a smaller subset of the very large CelebA dataset and contains roughly 30,000 images. 
# 
# Your first task consists in building the dataloader. To do so, you need to do the following:
# * implement the get_transforms function
# * create a custom Dataset class that reads the CelebA data

# ### Exercise: implement the get_transforms function
# 
# The `get_transforms` function should output a [`torchvision.transforms.Compose`](https://pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html#torchvision.transforms.Compose) of different transformations. You have two constraints:
# * the function takes a tuple of size as input and should **resize the images** to the input size
# * the output images should have values **ranging from -1 to 1**

# In[3]:


def get_transforms(size: Tuple[int, int]) -> Callable:
    """ Transforms to apply to the image."""
    # TODO: edit this function by appending transforms to the below list  
    transforms = [transform.ToTensor(),
                 transform.Resize(size),
                 transform.RandomHorizontalFlip(0.5),
                 transform.RandomPerspective(p=0.2),
                 transform.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
    
    return transform.Compose(transforms)


# ### Exercise: implement the DatasetDirectory class
# 
# 
# The `DatasetDirectory` class is a torch Dataset that reads from the above data directory. The `__getitem__` method should output a transformed tensor and the `__len__` method should output the number of files in our dataset. You can look at [this custom dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files) for ideas. 

# In[4]:


import os

class DatasetDirectory(Dataset):
    """
    A custom dataset class that loads images from folder.
    args:
    - directory: location of the images
    - transform: transform function to apply to the images
    - extension: file format
    """
    def __init__(self, 
                 directory: str, 
                 transforms: Callable = None, 
                 extension: str = '.jpg'):
        # TODO: implement the init method
        self._directory=directory
        self._transforms=transforms
        self._extension=extension
        self._images = os.listdir(data_dir)
        
        for item in self._images:
            if not item.endswith(self._extension):
                self._images.remove(item)

    def __len__(self) -> int:
        """ returns the number of items in the dataset """
        # TODO: return the number of elements in the dataset
        return len(self._images)

    def __getitem__(self, index: int) -> torch.Tensor:
        """ load an image and apply transformation """
        # TODO: return the index-element of the dataset
        filename = self._images[index]
        image = Image.open(os.path.join(self._directory, filename)).convert('RGB')
        if self._transforms != None:
            image = self._transforms(image)
        return image
        


# In[5]:


"""
DO NOT MODIFY ANYTHING IN THIS CELL
"""
# run this cell to verify your dataset implementation
dataset = DatasetDirectory(data_dir, get_transforms((64, 64)))
tests.check_dataset_outputs(dataset)


# The functions below will help you visualize images from the dataset.

# In[6]:


"""
DO NOT MODIFY ANYTHING IN THIS CELL
"""

def denormalize(images):
    """Transform images from [-1.0, 1.0] to [0, 255] and cast them to uint8."""
    return ((images + 1.) / 2. * 255).astype(np.uint8)

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(20, 4))
plot_size=20
for idx in np.arange(plot_size):
    ax = fig.add_subplot(2, int(plot_size/2), idx+1, xticks=[], yticks=[])
    img = dataset[idx].numpy()
    img = np.transpose(img, (1, 2, 0))
    img = denormalize(img)
    ax.imshow(img)


# ## Model implementation
# 
# As you know, a GAN is comprised of two adversarial networks, a discriminator and a generator. Now that we have a working data pipeline, we need to implement the discriminator and the generator. 
# 
# Feel free to implement any additional class or function.

# ### Exercise: Create the discriminator
# 
# The discriminator's job is to score real and fake images. You have two constraints here:
# * the discriminator takes as input a **batch of 64x64x3 images**
# * the output should be a single value (=score)
# 
# Feel free to get inspiration from the different architectures we talked about in the course, such as DCGAN, WGAN-GP or DRAGAN.
# 
# #### Some tips
# * To scale down from the input image, you can either use `Conv2d` layers with the correct hyperparameters or Pooling layers.
# * If you plan on using gradient penalty, do not use Batch Normalization layers in the discriminator.

# In[7]:


from torch.nn import Module
import torch.nn as nn


# In[8]:


class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # TODO: instantiate the different layers
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, padding=1, stride=2, bias=False),
            #nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, padding=1, stride=2, bias=False),
            #nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, padding=1, stride=2, bias=False),
            #nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, 4, padding=1, stride=2, bias=False),
            #nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(512, 1, 4, padding=0, stride=1, bias=False),
            nn.Sigmoid(),
            
            #nn.Flatten(),
            #nn.Dropout(0.4),
            
            #nn.Linear(8192, out_features=512),
            #nn.ReLU(),
            #nn.Linear(512, out_features=256),
            #nn.ReLU(),
            #nn.Linear(256, out_features=128),
            #nn.ReLU(),
            #nn.Linear(128, out_features=1),

        )
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement the forward method
         return self.model(x)  
    
    # https://github.com/nageshsinghc4/Face-generation-GAN/blob/master/face_GAN.ipynb


# In[9]:


"""
DO NOT MODIFY ANYTHING IN THIS CELL
"""
# run this cell to check your discriminator implementation
discriminator = Discriminator()
tests.check_discriminator(discriminator)


# ### Exercise: create the generator
# 
# The generator's job creates the "fake images" and learns the dataset distribution. You have three constraints here:
# * the generator takes as input a vector of dimension `[batch_size, latent_dimension, 1, 1]`
# * the generator must outputs **64x64x3 images**
# 
# Feel free to get inspiration from the different architectures we talked about in the course, such as DCGAN, WGAN-GP or DRAGAN.
# 
# #### Some tips:
# * to scale up from the latent vector input, you can use `ConvTranspose2d` layers
# * as often with Gan, **Batch Normalization** helps with training

# In[10]:


class Generator(Module):
    def __init__(self, latent_dim: int):
        super(Generator, self).__init__()
        # TODO: instantiate the different layers
        self._latent_dim = latent_dim
        
        self.model = nn.Sequential(
            nn.ConvTranspose2d(self._latent_dim, 512, 4, padding=0, stride=1, bias=False),
            #nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.ConvTranspose2d(512, 256, 4, padding=1, stride=2, bias=False),
            #nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, 4, padding=1, stride=2, bias=False),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 4, padding=1, stride=2, bias=False),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 3, 4, padding=1, stride=2, bias=False),
            nn.Tanh()
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement the forward method
        return self.model(x)  


# In[11]:


"""
DO NOT MODIFY ANYTHING IN THIS CELL
"""
# run this cell to verify your generator implementation
latent_dim = 128
generator = Generator(latent_dim)
tests.check_generator(generator, latent_dim)


# ## Optimizer
# 
# In the following section, we create the optimizers for the generator and discriminator. You may want to experiment with different optimizers, learning rates and other hyperparameters as they tend to impact the output quality.

# ### Exercise: implement the optimizers

# In[12]:


import torch.optim as optim


def create_optimizers(generator: Module, discriminator: Module):
    """ This function should return the optimizers of the generator and the discriminator """
    # TODO: implement the generator and discriminator optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    return g_optimizer, d_optimizer


# ## Losses implementation
# 
# In this section, we are going to implement the loss function for the generator and the discriminator. You can and should experiment with different loss function.
# 
# Some tips:
# * You can choose the commonly used the binary cross entropy loss or select other losses we have discovered in the course, such as the Wasserstein distance.
# * You may want to implement a gradient penalty function as discussed in the course. It is not required and the code will work whether you implement it or not.

# ### Exercise: implement the generator loss
# 
# The generator's goal is to get the discriminator to think its generated images (= "fake" images) are real.

# In[13]:


def generator_loss(fake_logits):
    """ Generator loss, takes the fake scores as inputs. """
    # TODO: implement the generator loss
    loss = -fake_logits.mean()
    
    return loss


# ### Exercise: implement the discriminator loss
# 
# We want the discriminator to give high scores to real images and low scores to fake ones and the discriminator loss should reflect that.

# In[14]:


def discriminator_loss(real_logits, fake_logits):
    """ Discriminator loss, takes the fake and real logits as inputs. """
    # TODO: implement the discriminator loss 
    real_loss = -real_logits.mean()
    fake_loss = fake_logits.mean()
    
    return real_loss + fake_loss


# ### Exercise (Optional): Implement the gradient Penalty
# 
# In the course, we discussed the importance of gradient penalty in training certain types of Gans. Implementing this function is not required and depends on some of the design decision you made (discriminator architecture, loss functions).

# In[15]:


# Dummy critic class for gradient penalty
class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.pow(x, 2)


# In[16]:


def gradient_penalty(discriminator, real_sample, fake_sample):
    """ This function enforces """
    gp = 0
    # TODO (Optional): implement the gradient penalty
    
    # sample a random point between both distributions
    alpha = torch.rand(real_sample.shape).to(device)
    #real_sample = real_sample.to(device)
    x_hat = alpha * real_sample + (1 - alpha) * fake_sample
    #x_hat = x_hat.to(device)
    
    # calculate the gradient
    x_hat.requires_grad = True
    pred = discriminator(x_hat)
    grad = torch.autograd.grad(pred, 
                               x_hat, 
                               grad_outputs=torch.ones_like(pred), 
                               create_graph=True)[0]
    
    # calculate the norm and the final penalty
    norm = torch.norm(grad.view(-1), 2)
    gp = ((norm - 1)**2).mean()  
    
    return gp


# ## Training
# 
# 
# Training will involve alternating between training the discriminator and the generator. You'll use your functions real_loss and fake_loss to help you calculate the discriminator losses.
# 
# * You should train the discriminator by alternating on real and fake images
# * Then the generator, which tries to trick the discriminator and should have an opposing loss function

# ### Exercise: implement the generator step and the discriminator step functions
# 
# Each function should do the following:
# * calculate the loss
# * backpropagate the gradient
# * perform one optimizer step

# In[35]:


def generator_step(batch_size: int, latent_dim: int) -> Dict:
    """ One training step of the generator. """
    # TODO: implement the generator step (foward pass, loss calculation and backward pass)
    g_optimizer.zero_grad()
    
    z = np.random.uniform(-1, 1, size=(batch_size,latent_dim, 1, 1))
    z = torch.from_numpy(z).float()#.to(device)
    
    train_on_gpu = torch.cuda.is_available()
    
    if train_on_gpu:
        z = z.cuda()
    
    fake_images = generator(z)
    d_fake = discriminator(fake_images)
    
    g_loss = generator_loss(d_fake)
    
    g_loss.backward()
    g_optimizer.step()
    
    return {'loss': g_loss}


def discriminator_step(batch_size: int, latent_dim: int, real_images: torch.Tensor) -> Dict:
    """ One training step of the discriminator. """
    # TODO: implement the discriminator step (foward pass, loss calculation and backward pass)
    d_optimizer.zero_grad()
    
    train_on_gpu = torch.cuda.is_available()
    
    if train_on_gpu:
        real_images = real_images.cuda()
        discriminator.cuda()
        generator.cuda()
        
    d_real = discriminator(real_images)
    
    z = np.random.uniform(-1, 1, size=(batch_size, latent_dim, 1, 1))
    z = torch.from_numpy(z).float()#.to(device)
    
    if train_on_gpu:
        z = z.cuda()
    
    with torch.no_grad(): # recommendation from a mentor
        fake_images = generator(z)
    #fake_images = fake_images.detach() # recommendation from a mentor
    
    d_fake = discriminator(fake_images.detach())
    d_fake.retain_grad() # recommendation from a mentor 
    
    gp = gradient_penalty(Critic(), d_real, d_fake)#.to(device)
    
    d_loss = discriminator_loss(real_images, fake_images) + gp
    
    d_loss.backward()
    d_optimizer.step()
        
    return {'loss': d_loss, 'gp': gp}


# ### Main training loop
# 
# You don't have to implement anything here but you can experiment with different hyperparameters.

# In[36]:


from datetime import datetime


# In[37]:


# you can experiment with different dimensions of latent spaces
latent_dim = 128

# update to cpu if you do not have access to a gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# number of epochs to train your model
n_epochs = 1

# number of images in each batch
batch_size = 64


# In[38]:


"""
DO NOT MODIFY ANYTHING IN THIS CELL
"""
print_every = 50

# Create optimizers for the discriminator D and generator G
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)
g_optimizer, d_optimizer = create_optimizers(generator, discriminator)

dataloader = DataLoader(dataset, 
                        batch_size=64, 
                        shuffle=True, 
                        num_workers=4, 
                        drop_last=True,
                        pin_memory=False)


# In[39]:


"""
DO NOT MODIFY ANYTHING IN THIS CELL
"""

def display(fixed_latent_vector: torch.Tensor):
    """ helper function to display images during training """
    fig = plt.figure(figsize=(14, 4))
    plot_size = 16
    for idx in np.arange(plot_size):
        ax = fig.add_subplot(2, int(plot_size/2), idx+1, xticks=[], yticks=[])
        img = fixed_latent_vector[idx, ...].detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = denormalize(img)
        ax.imshow(img)
    plt.show()


# ### Exercise: implement the training strategy
# 
# You should experiment with different training strategies. For example:
# 
# * train the generator more often than the discriminator. 
# * added noise to the input image
# * use label smoothing
# 
# Implement with your training strategy below.

# In[40]:


fixed_latent_vector = torch.randn(16, latent_dim, 1, 1).float().to(device)

losses = []
for epoch in range(n_epochs):
    for batch_i, real_images in enumerate(dataloader):
        real_images = real_images.to(device)
        
        ####################################
        
        # TODO: implement the training strategy
    
        ####################################
        
        # Discriminator
        d_loss = discriminator_step(batch_size, latent_dim, real_images)
        # Generator
        g_loss = generator_step(batch_size, latent_dim)
        
        if batch_i % print_every == 0:
            # append discriminator loss and generator loss
            d = d_loss['loss'].item()
            g = g_loss['loss'].item() 
            losses.append((d, g))
            # print discriminator and generator loss
            time = str(datetime.now()).split('.')[0]
            print(f'{time} | Epoch [{epoch+1}/{n_epochs}] | Batch {batch_i}/{len(dataloader)} | d_loss: {d:.4f} | g_loss: {g:.4f}')
    
    # display images during training
    generator.eval()
    generated_images = generator(fixed_latent_vector)
    display(generated_images)
    generator.train()


# ### Training losses
# 
# Plot the training losses for the generator and discriminator.

# In[ ]:


"""
DO NOT MODIFY ANYTHING IN THIS CELL
"""
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
plt.plot(losses.T[1], label='Generator', alpha=0.5)
plt.title("Training Losses")
plt.legend()


# ### Question: What do you notice about your generated samples and how might you improve this model?
# When you answer this question, consider the following factors:
# * The dataset is biased; it is made of "celebrity" faces that are mostly white
# * Model size; larger models have the opportunity to learn more features in a data feature space
# * Optimization strategy; optimizers and number of epochs affect your final result
# * Loss functions

# **Answer:** (Write your answer in this cell)

# ### Submitting This Project
# When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_face_generation.ipynb".  
# 
# Submit the notebook using the ***SUBMIT*** button in the bottom right corner of the Project Workspace.

# In[ ]:




