# import torch
# import os
# from sevae.networks.VAE.vae import VAE, Encoder, ImgDecoder

# # Paths
# CHECKPOINT_PATH = "../../weights/seVAE_weights/seVAE_evo_LD_128.pth"  # your trained weights
# EXPORT_DIR = "onnx_exports"
# os.makedirs(EXPORT_DIR, exist_ok=True)

# # Model settings
# LATENT_DIM = 128
# INPUT_CHANNELS = 1
# HEIGHT, WIDTH = 270, 480

# # Load model
# vae = VAE(input_dim=INPUT_CHANNELS, latent_dim=LATENT_DIM)
# state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
# vae.load_state_dict(state_dict)
# vae.eval().to("cuda")

# encoder = vae.encode
# decoder = vae.img_decoder

# # Dummy inputs
# dummy_img = torch.randn(1, 1, HEIGHT, WIDTH, device="cuda")
# dummy_latent = torch.randn(1, LATENT_DIM, device="cuda")

# # Export encoder
# encoder_path = os.path.join(EXPORT_DIR, "vae_encoder.onnx")
# torch.onnx.export(
#     encoder,
#     dummy_img,
#     encoder_path,
#     input_names=["input_image"],
#     output_names=["latent_vector"],
#     opset_version=13,
#     dynamic_axes={"input_image": {0: "batch"}, "latent_vector": {0: "batch"}}
# )
# print(f"✅ Exported encoder to {encoder_path}")

# # Export decoder
# decoder_path = os.path.join(EXPORT_DIR, "vae_decoder.onnx")
# torch.onnx.export(
#     decoder,
#     dummy_latent,
#     decoder_path,
#     input_names=["latent"],
#     output_names=["reconstruction"],
#     opset_version=13,
#     dynamic_axes={"latent": {0: "batch"}, "reconstruction": {0: "batch"}}
# )
# print(f"✅ Exported decoder to {decoder_path}")

import torch
import os
from sevae.networks.VAE.vae import VAE, ImgDecoder

# Paths
CHECKPOINT_PATH = "../../weights/seVAE_weights/seVAE_evo_LD_128.pth"
EXPORT_DIR = "onnx_exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

# Model settings
LATENT_DIM = 128
INPUT_CHANNELS = 1
HEIGHT, WIDTH = 270, 480

# Load VAE
vae = VAE(input_dim=INPUT_CHANNELS, latent_dim=LATENT_DIM)
state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
vae.load_state_dict(state_dict)
vae.eval().to("cuda")

# -------------------------
# Encoder wrapper for ONNX
# -------------------------
class EncoderWrapper(torch.nn.Module):
    def __init__(self, vae_model):
        super().__init__()
        self.encoder = vae_model.encoder
        self.latent_dim = vae_model.latent_dim
        # extract mean from latent vector
        self.mean_params = lambda x: x[:, :self.latent_dim]

    def forward(self, x):
        z = self.encoder(x)
        mean = self.mean_params(z)
        return mean

encoder_model = EncoderWrapper(vae).to("cuda")
dummy_img = torch.randn(1, INPUT_CHANNELS, HEIGHT, WIDTH, device="cuda")

encoder_path = os.path.join(EXPORT_DIR, "vae_encoder.onnx")
torch.onnx.export(
    encoder_model,
    dummy_img,
    encoder_path,
    input_names=["input_image"],
    output_names=["latent_vector"],
    opset_version=13,
    dynamic_axes={"input_image": {0: "batch"}, "latent_vector": {0: "batch"}},
)
print(f"✅ Exported encoder to {encoder_path}")

# -------------------------
# Decoder
# -------------------------
decoder_model = vae.img_decoder.to("cuda")
dummy_latent = torch.randn(1, LATENT_DIM, device="cuda")

decoder_path = os.path.join(EXPORT_DIR, "vae_decoder.onnx")
torch.onnx.export(
    decoder_model,
    dummy_latent,
    decoder_path,
    input_names=["latent_vector"],
    output_names=["reconstruction"],
    opset_version=13,
    dynamic_axes={"latent_vector": {0: "batch"}, "reconstruction": {0: "batch"}},
)
print(f"✅ Exported decoder to {decoder_path}")

