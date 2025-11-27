import torch
import torch.nn as nn

class ImgDecoder(nn.Module):
    """Generates an image reconstruction using ConvTranspose2d layers. 
    
    Adapted from https://github.com/microsoft/AirSim-Drone-Racing-VAE-Imitation/blob/master/racing_models/cmvae.py
    """

    def __init__(self, input_dim=1, latent_dim=64, with_logits=False):
        """
        Parameters
        ----------
        latent_dim: int
            The latent dimension.
        """

        super(ImgDecoder, self).__init__()
        self.with_logits = with_logits
        self.n_channels = input_dim
        self.dense = nn.Linear(latent_dim, 512)
        self.dense1 = nn.Linear(512, 12*17*128)
        
        # Pytorch docs: output_padding is only used to find output shape, but does not actually add zero-padding to output
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=8, stride=2, padding=(2,2), output_padding=(1,1), dilation=1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=8, stride=4, padding=(2,2), output_padding=(0,0), dilation=1)
        self.deconv6 = nn.ConvTranspose2d(32, 16, kernel_size=9, stride=2, padding=(0,0), output_padding=(1,1))
        self.deconv7 = nn.ConvTranspose2d(16, self.n_channels, kernel_size=6, stride=2, padding=2) # tanh activation or sigmoid

        print('[ImgDecoder] Done with create_model')

        print("Defined decoder.")

    def forward(self, z):
        return self.decode(z)
    
    def decode(self, z):
        x = self.dense(z)
        x = torch.relu(x)
        x = self.dense1(x)
        x = x.view(x.size(0), 128, 12, 17)

        x = self.deconv1(x)
        x = torch.relu(x)
        #print(x.shape)
        x = self.deconv2(x)
        x = torch.relu(x)
        #print(x.shape)
        x = self.deconv4(x)
        x = torch.relu(x)
        #print(x.shape)

        x = self.deconv6(x)
        x = torch.relu(x)
        #print(x.shape)
        x = self.deconv7(x)
        #print(x.shape)
        # print(f"- After deconv 7, mean: {x.mean():.3f} var: {x.var():.3f}")
        if self.with_logits:
            return x

        x = torch.sigmoid(x)
        # print(f"- After sigmoid, mean: {x.mean():.3f} var: {x.var():.3f}")
        return x


    #     super(ImgDecoder, self).__init__()
    #     print('[ImgDecoder] Starting create_model')
    #     self.with_logits = with_logits
    #     self.n_channels = input_dim
    #     # self.dense = nn.Linear(latent_dim, 512)
    #     # self.dense1 = nn.Linear(512, 9*15*128)
        
    #     # # Pytorch docs: output_padding is only used to find output shape, but does not actually add zero-padding to output
    #     # self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
    #     # self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=(2,2), output_padding=(0,1), dilation=1)
    #     # self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=4, padding=(2,2), output_padding=(0,0), dilation=1)
    #     # self.deconv6 = nn.ConvTranspose2dls, kernel_size=4, stride=2, padding=2) # tanh activation or sigmoid

    #     self.dense = nn.Linear(latent_dim, 512)
    #     self.dense1 = nn.Linear(512, 128 * 30 * 40)  # Adjusted to output 128 x 30 x 40 tensor
        
    #     # ConvTranspose2d layers
    #     # self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
    #     # self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=(2, 2), output_padding=(0, 0), dilation=1)
    #     # self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=(2, 2), output_padding=(0, 0), dilation=1)  # Adjusted to output 32 x 120 x 160 tensor
    #     # self.deconv6 = nn.ConvTranspose2d(32, 16, kernel_size=6, stride=2, padding=(2, 2), output_padding=(0, 0), dilation=1)  # Adjusted to output 16 x 240 x 320 tensor
    #     # self.deconv7 = nn.ConvTranspose2d(16, self.n_channels, kernel_size=6, stride=2, padding=(2, 2), output_padding=(0, 0), dilation=1)  # Adjusted to output 1 x 480 x 640 tensor
    #     self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
    #     self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=(1, 1), output_padding=(0, 0), dilation=1)
    #     self.deconv3 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=(1, 1), output_padding=(1, 1), dilation=1)
    #     self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=(1, 1), output_padding=(1, 1), dilation=1)  # Adjusted to output 32 x 120 x 160 tensor
    #     self.deconv5 = nn.ConvTranspose2d(32, 32, kernel_size=6, stride=2, padding=(1, 1), output_padding=(1, 1), dilation=1)
    #     self.deconv6 = nn.ConvTranspose2d(32, 16, kernel_size=6, stride=2, padding=(1, 1), output_padding=(1, 1), dilation=1)  # Adjusted to output 16 x 240 x 320 tensor
    #     self.deconv7 = nn.ConvTranspose2d(16, self.n_channels, kernel_size=6, stride=2, padding=(1, 1), output_padding=(0, 0), dilation=1)  # Adjusted to output 1 x 480 x 640 tensor
    #     print('[ImgDecoder] Done with create_model')

    #     print("Defined decoder.")

    # def forward(self, z):
    #     return self.decode(z)
    
    # def decode(self, z):
    #     print(z.shape)
    #     x = self.dense(z)
    #     x = self.dense(z)
    #     x = torch.relu(x)
    #     print(x.shape)
    #     x = self.dense1(x)
    #     print(x.shape)
    #     x = x.view(x.size(0), 128, 30, 40)
    #     print(x.shape)
    #     x = self.deconv1(x)
    #     x = torch.relu(x)
    #     print("dio letame")
    #     x = self.deconv2(x)
    #     x = torch.relu(x)
    #     print("dio letame")
    #     x = self.deconv3(x)
    #     x = torch.relu(x)
    #     print("dio letame")
    #     x = self.deconv4(x)
    #     x = torch.relu(x)
    #     print("dio letame")
    #     x = self.deconv5(x)
    #     x = torch.relu(x)
    #     print("dio letame")

    #     x = self.deconv6(x)
    #     x = torch.relu(x)
    #     print("dio letame")
    #     x = self.deconv7(x)
    #     print(x.shape)
    #     # print(f"- After deconv 7, mean: {x.mean():.3f} var: {x.var():.3f}")
    #     if self.with_logits:
    #         print(x.shape)
    #         return x
    #     x = torch.sigmoid(x)
    #     # print(f"- After sigmoid, mean: {x.mean():.3f} var: {x.var():.3f}")
    #     print(x.shape)
    #     return x



if __name__== '__main__':
    from torchsummary import summary
    
    latent_dim = 64
    img_decoder = ImgDecoder(latent_dim=latent_dim, input_dim=1).to("cuda")
    summary(img_decoder, (1,latent_dim), device="cuda")