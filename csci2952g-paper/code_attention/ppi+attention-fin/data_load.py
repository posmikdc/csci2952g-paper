import torch

# Load the file
try:
    data = torch.load('/users/cnaraya2/ppi/HIGH-PPI/protein_info/x_list.pt', weights_only=False) # Use your actual file name
except Exception as e:
    print(f"Error loading file: {e}")
    # If loading fails, the file may be corrupt or was saved with an incompatible PyTorch version.
    # Re-run your data generation script.
    exit()

print(f"--- File Content Summary ---")
print(f"Overall Data Type: {type(data)}")
print(f"Number of Proteins (Elements in the List): {len(data)}")

# Inspect the first few elements
for i in range(min(5, len(data))):
    p = data[i]
    print(f"\n--- Protein {i} ---")
    print(f"Data Type: {type(p)}")
    
    # Check if it's a tensor or NumPy array (the common issue)
    if isinstance(p, torch.Tensor):
        print(f"Shape: {p.shape}")
        print(f"Datatype (Dtype): {p.dtype}")
    elif isinstance(p, torch.Tensor):
        # Should now be correct after the NumPy fix, but check for clarity
        print(f"Shape: {p.shape}")
        print(f"Datatype (Dtype): {p.dtype}")
    elif hasattr(p, 'shape'):
        # This catches NumPy arrays if the fix wasn't correctly implemented
        print(f"Shape (NumPy): {p.shape}")
    else:
        print("Element is not a tensor or array (unexpected format).")