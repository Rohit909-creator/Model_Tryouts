import torch
import pandas as pd


def smape_loss(predicted_price, actual_price, epsilon=1e-8):
    """
    SMAPE loss function for training.
    
    Args:
        predicted_price: torch.Tensor of predictions
        actual_price: torch.Tensor of actual values
        epsilon: small constant to avoid division by zero
    
    Returns:
        SMAPE score (lower is better)
    """
    numerator = torch.abs(predicted_price - actual_price)
    denominator = (torch.abs(predicted_price) + torch.abs(actual_price)) / 2
    
    # Add epsilon to avoid division by zero
    smape_score = torch.mean(numerator / (denominator + epsilon))
    
    return smape_score

# Read the CSV file
df = pd.read_csv(r"C:\Users\Rohit Francis\Downloads\test_out.csv")
print(df.head())

# Extract prices
actual_prices = df['price'].values

# For demonstration, let's create some dummy predictions
# Replace this with your actual model predictions
predicted_prices = actual_prices * 0.9  # Dummy predictions (10% off)

# Convert to tensors
actual_tensor = torch.tensor(actual_prices, dtype=torch.float32)
predicted_tensor = torch.tensor(predicted_prices, dtype=torch.float32)

# Calculate SMAPE for every 25000 samples
chunk_size = 25000
total_samples = len(actual_tensor)
num_chunks = (total_samples + chunk_size - 1) // chunk_size  # Ceiling division

print(f"Total samples: {total_samples}")
print(f"Number of chunks: {num_chunks}")
print(f"Chunk size: {chunk_size}\n")

smape_scores = []

for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, total_samples)
    
    # Get chunk
    actual_chunk = actual_tensor[start_idx:end_idx]
    predicted_chunk = predicted_tensor[start_idx:end_idx]
    
    # Calculate SMAPE for this chunk
    chunk_smape = smape_loss(predicted_chunk, actual_chunk)
    smape_scores.append(chunk_smape.item())
    
    print(f"Chunk {i+1} (samples {start_idx:6d} to {end_idx:6d}): SMAPE = {chunk_smape.item():.6f}")

# Calculate overall SMAPE
overall_smape = smape_loss(predicted_tensor, actual_tensor)
print(f"\nOverall SMAPE (all {total_samples} samples): {overall_smape.item():.6f}")
print(f"Average of chunk SMAPEs: {sum(smape_scores)/len(smape_scores):.6f}")
# # Test cases
# print(smape_loss(torch.tensor([10.12]), torch.tensor([10.12])))  # ~0.0
# print(smape_loss(torch.tensor([10.0, 20.0]), torch.tensor([12.0, 18.0])))  # Some error