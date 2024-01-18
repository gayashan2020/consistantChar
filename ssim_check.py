import torch
import torchvision.transforms as transforms
from PIL import Image
from piq import ssim, SSIMLoss

# Load PNG images
image1_path = '/content/t3.png'
image2_path = '/content/gen3.PNG'

image1 = Image.open(image1_path).convert('RGB')
image2 = Image.open(image2_path).convert('RGB')

# Ensure both images are of the same size (in this example, 256x256)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

x = transform(image1).unsqueeze(0)  # Add batch dimension
y = transform(image2).unsqueeze(0)  # Add batch dimension

# Ensure requires_grad is set to True for x
x.requires_grad_(True)

# Compute SSIM index
ssim_index: torch.Tensor = ssim(x, y, data_range=1.)

# Compute SSIM loss
loss = SSIMLoss(data_range=1.)
output: torch.Tensor = loss(x, y)
output.backward()

print(f"SSIM Index: {ssim_index.item()}")
print(f"SSIM Loss: {output.item()}")