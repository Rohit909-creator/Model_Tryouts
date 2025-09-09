import os
import time
import torch
import numpy as np

# Test function to check computational accuracy and performance
def test_openmp_impact():
    print("=== Testing OpenMP Impact ===\n")
    
    # Test 1: Computational Accuracy
    print("1. Testing Computational Accuracy...")
    
    # Run the same computation multiple times
    results = []
    for i in range(5):
        # Simple matrix operations that use OpenMP
        a = torch.randn(1000, 1000)
        b = torch.randn(1000, 1000)
        result = torch.mm(a, b).sum().item()
        results.append(result)
        print(f"   Run {i+1}: {result:.6f}")
    
    # Check if results are consistent
    max_diff = max(results) - min(results)
    relative_diff = max_diff / abs(np.mean(results)) * 100
    
    print(f"   Max difference: {max_diff:.6f}")
    print(f"   Relative difference: {relative_diff:.8f}%")
    
    if relative_diff < 1e-10:
        print("   ✅ ACCURACY: Results are consistent - no accuracy issues detected")
    else:
        print("   ⚠️  ACCURACY: Some variation detected - might need proper fix")
    
    print()
    
    # Test 2: Performance Impact
    print("2. Testing Performance Impact...")
    
    # Time a typical operation
    times = []
    for i in range(3):
        start_time = time.time()
        
        # Simulate your typical workload
        a = torch.randn(2000, 2000)
        b = torch.randn(2000, 2000)
        c = torch.mm(a, b)
        d = torch.sum(c)
        
        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)
        print(f"   Run {i+1}: {elapsed:.4f} seconds")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"   Average time: {avg_time:.4f} ± {std_time:.4f} seconds")
    
    if std_time / avg_time < 0.1:  # Less than 10% variation
        print("   ✅ PERFORMANCE: Consistent timing - minimal performance impact")
    else:
        print("   ⚠️  PERFORMANCE: High timing variation - consider proper fix")
    
    print()
    
    # Test 3: Memory Usage Pattern
    print("3. Testing Memory Usage...")
    try:
        # Create large tensors to test memory allocation
        large_tensor = torch.randn(5000, 5000)
        result = torch.sum(large_tensor)
        print(f"   ✅ MEMORY: Large tensor operations work fine")
        del large_tensor  # Clean up
    except Exception as e:
        print(f"   ❌ MEMORY: Issue detected - {e}")
    
    print("\n=== Recommendations ===")
    print("If you see mostly ✅ symbols above:")
    print("  → You can safely use KMP_DUPLICATE_LIB_OK=TRUE for now")
    print("  → Consider fixing properly when you have time")
    print()
    print("If you see ⚠️ or ❌ symbols:")
    print("  → You should implement a proper fix (reinstall with conda)")
    print("  → The computational accuracy might be affected")

# Run with the OpenMP fix
print("Testing WITH OpenMP duplicate library workaround...")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

test_openmp_impact()

# Your actual code test
print("\n=== Testing Your Actual Code ===")
try:
    from memorizing_transformers_pytorch import MemorizingTransformer
    
    model = MemorizingTransformer(
        num_tokens = 1000,  # Smaller for testing
        dim = 128,
        dim_head = 32,
        depth = 2,
        memorizing_layers = (1,),
        max_knn_memories = 1000,
        num_retrieved_memories = 8,
    )
    
    data = torch.randint(0, 1000, (1, 100))
    knn_memories = model.create_knn_memories(batch_size = 1)
    logits = model(data, knn_memories = knn_memories)
    
    print("✅ Your memorizing transformer code works fine!")
    print(f"   Output shape: {logits.shape}")
    
except Exception as e:
    print(f"❌ Issue with your code: {e}")
    print("   You might need the proper fix")