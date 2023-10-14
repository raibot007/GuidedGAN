import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels, 
                           kernel_size, stride, padding, bias=False)
    
    # append conv layer
    layers.append(conv_layer)

    if batch_norm:
        # append batchnorm layer
        layers.append(nn.BatchNorm2d(out_channels))
     
    # using Sequential container
    return nn.Sequential(*layers)
    
# helper deconv function
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    transpose_conv_layer = nn.ConvTranspose2d(in_channels, out_channels, 
                                              kernel_size, stride, padding, bias=False)
    # append transpose convolutional layer
    layers.append(transpose_conv_layer)
    
    if batch_norm:
        # append batchnorm layer
        layers.append(nn.BatchNorm2d(out_channels))
        
    return nn.Sequential(*layers)

# Define a class for your GAN generator and discriminator
class Generator(nn.Module):
    # Define generator architecture for a specific timestep    
    def __init__(self, z_size, conv_dim=32):
        """
        Initialize the Generator Module
        :param z_size: The length of the input latent vector, z
        :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer
        """
        super(Generator, self).__init__()

        # complete init function
        self.conv_dim = conv_dim
        self.fc = nn.Linear(z_size, conv_dim*8*2*2)
        self.t_conv1 = deconv(conv_dim*8, conv_dim*4, 4)
        self.t_conv2 = deconv(conv_dim*4, conv_dim*2, 4)
        self.t_conv3 = deconv(conv_dim*2, conv_dim, 4)
        self.t_conv4 = deconv(conv_dim, 3, 4, batch_norm=False)

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: A 32x32x3 Tensor image as output
        """
        # define feedforward behavior
        out = self.fc(x)
        out = out.view(-1, self.conv_dim*8, 2, 2)
        out = F.relu(self.t_conv1(out))
        out = F.relu(self.t_conv2(out))
        out = F.relu(self.t_conv3(out))
        out = self.t_conv4(out)
        out = F.tanh(out)
        
        return out


class Discriminator(nn.Module):
    # Define discriminator architecture for a specific timestep

    def __init__(self, conv_dim=32):
        """
        Initialize the Discriminator Module
        :param conv_dim: The depth of the first convolutional layer
        """
        super(Discriminator, self).__init__()

        # complete init function
        self.conv_dim = conv_dim
        self.conv1 = conv(3, conv_dim, 4, batch_norm=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4)
        self.fc = nn.Linear(conv_dim*8*2*2, 1)

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: Discriminator logits; the output of the neural network
        """
        # define feedforward behavior
        out = F.leaky_relu(self.conv1(x), 0.2)
        out = F.leaky_relu(self.conv2(out), 0.2)
        out = F.leaky_relu(self.conv3(out), 0.2)
        out = F.leaky_relu(self.conv4(out), 0.2)
        out = out.view(-1, self.conv_dim*8*2*2)
        out = self.fc(out)
        
        return out


# Define hyperparameters
num_timesteps = 4
latent_dim = 100
batch_size = 64

# Initialize the GANs for each timestep
generators = [Generator() for _ in range(num_timesteps)]
discriminators = [Discriminator() for _ in range(num_timesteps)]

# Define optimizers for each GAN
gen_optimizers = [optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999)) for generator in generators]
dis_optimizers = [optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)) for discriminator in discriminators]

# Training loop
for timestep in range(num_timesteps):
    for epoch in range(num_epochs):
        for batch in data_loader:
            # Train the discriminator
            real_data = batch.to(device)
            fake_data = generators[timestep](generate_latent_noise(batch_size, latent_dim)).detach()
            dis_loss = compute_discriminator_loss(discriminators[timestep], real_data, fake_data)
            dis_optimizers[timestep].zero_grad()
            dis_loss.backward()
            dis_optimizers[timestep].step()

            # Train the generator
            fake_data = generators[timestep](generate_latent_noise(batch_size, latent_dim))
            gen_loss = compute_generator_loss(discriminators[timestep], fake_data)
            gen_optimizers[timestep].zero_grad()
            gen_loss.backward()
            gen_optimizers[timestep].step()

    # Optionally, update the latent noise or generator architecture for the next timestep

# Combine the outputs of each timestep to get the final result
final_image = combine_timestep_outputs(generators)

# Save or display the final result
