import torch

# Load the FER+ checkpoint
checkpoint = torch.load('models/ferplus_checkpoints/best_model.pt', map_location='cpu', weights_only=False)

print("Checkpoint keys:", checkpoint.keys())
print("\n" + "="*50)

if 'model_state' in checkpoint:
    state_dict = checkpoint['model_state']
elif 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
elif 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

print("\nFirst 20 model keys:")
for i, key in enumerate(list(state_dict.keys())[:20]):
    print(f"  {key}: {state_dict[key].shape}")

print("\n" + "="*50)
print(f"\nTotal number of keys: {len(state_dict.keys())}")

# Check for specific patterns
has_classifier = any('classifier' in k for k in state_dict.keys())
has_fc = any('fc' in k for k in state_dict.keys())
has_head = any('head' in k for k in state_dict.keys())

print(f"\nHas 'classifier' layers: {has_classifier}")
print(f"Has 'fc' layers: {has_fc}")
print(f"Has 'head' layers: {has_head}")

# Print last few keys (usually the classifier)
print("\nLast 10 model keys:")
for key in list(state_dict.keys())[-10:]:
    print(f"  {key}: {state_dict[key].shape}")
