from model_configs import MODEL_CONFIGS
from unet import UNetModel

# architecture
architecture = "mnist"
model = UNetModel(**MODEL_CONFIGS[architecture])