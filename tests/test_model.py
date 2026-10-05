import torch
from src.model import create_resnet18

model = create_resnet18(num_classes=2)

print(model)

dummy_input = torch.randn(16, 3, 224, 224)

output = model(dummy_input)

print("\nInput shape:")
print(dummy_input.shape)

print("\nOutput shape:")
print(output.shape)