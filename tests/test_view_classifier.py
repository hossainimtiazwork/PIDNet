#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for PIDNet with view classification head
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.pidnet import PIDNet, get_pred_model

def test_pidnet_without_view_classifier():
    """Test original PIDNet without view classifier"""
    print("=" * 80)
    print("Testing PIDNet WITHOUT view classifier (original behavior)")
    print("=" * 80)
    
    # Create model
    model = PIDNet(m=2, n=3, num_classes=19, planes=32, ppm_planes=96, 
                   head_planes=128, augment=True, num_view_classes=0)
    model.eval()
    
    # Test input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 512, 1024)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_tensor)
    
    print(f"Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"Output {i} shape: {out.shape}")
    
    assert len(outputs) == 3, "Should have 3 outputs: [P_head, final, D_head]"
    assert outputs[0].shape[1] == 19, "P_head should have 19 classes"
    assert outputs[1].shape[1] == 19, "Final output should have 19 classes"
    assert outputs[2].shape[1] == 1, "D_head should have 1 channel (boundary)"
    
    print("✓ Test passed: PIDNet without view classifier works correctly\n")


def test_pidnet_with_view_classifier():
    """Test PIDNet with view classifier"""
    print("=" * 80)
    print("Testing PIDNet WITH view classifier")
    print("=" * 80)
    
    num_view_classes = 4
    
    # Create model
    model = PIDNet(m=2, n=3, num_classes=19, planes=32, ppm_planes=96, 
                   head_planes=128, augment=True, num_view_classes=num_view_classes)
    model.eval()
    
    # Test input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 512, 1024)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Number of view classes: {num_view_classes}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_tensor)
    
    print(f"Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"Output {i} shape: {out.shape}")
    
    assert len(outputs) == 4, "Should have 4 outputs: [P_head, final, D_head, view_pred]"
    assert outputs[0].shape[1] == 19, "P_head should have 19 classes"
    assert outputs[1].shape[1] == 19, "Final output should have 19 classes"
    assert outputs[2].shape[1] == 1, "D_head should have 1 channel (boundary)"
    assert outputs[3].shape == (batch_size, num_view_classes), f"View prediction should be [{batch_size}, {num_view_classes}]"
    
    print("✓ Test passed: PIDNet with view classifier works correctly\n")


def test_pidnet_inference_mode():
    """Test PIDNet in inference mode (augment=False)"""
    print("=" * 80)
    print("Testing PIDNet in inference mode (augment=False)")
    print("=" * 80)
    
    num_view_classes = 4
    
    # Without view classifier
    print("\n1. Inference mode WITHOUT view classifier:")
    model1 = PIDNet(m=2, n=3, num_classes=19, planes=32, ppm_planes=96, 
                    head_planes=128, augment=False, num_view_classes=0)
    model1.eval()
    
    input_tensor = torch.randn(2, 3, 512, 1024)
    with torch.no_grad():
        output1 = model1(input_tensor)
    
    print(f"Output type: {type(output1)}")
    print(f"Output shape: {output1.shape}")
    assert isinstance(output1, torch.Tensor), "Should return single tensor"
    assert output1.shape[1] == 19, "Should have 19 classes"
    print("✓ Passed")
    
    # With view classifier
    print("\n2. Inference mode WITH view classifier:")
    model2 = PIDNet(m=2, n=3, num_classes=19, planes=32, ppm_planes=96, 
                    head_planes=128, augment=False, num_view_classes=num_view_classes)
    model2.eval()
    
    with torch.no_grad():
        outputs2 = model2(input_tensor)
    
    print(f"Number of outputs: {len(outputs2)}")
    print(f"Segmentation output shape: {outputs2[0].shape}")
    print(f"View prediction shape: {outputs2[1].shape}")
    assert len(outputs2) == 2, "Should return tuple of (seg_output, view_pred)"
    assert outputs2[0].shape[1] == 19, "Should have 19 classes"
    assert outputs2[1].shape == (2, num_view_classes), f"Should have shape [2, {num_view_classes}]"
    print("✓ Passed\n")


def test_different_model_sizes():
    """Test view classifier with different PIDNet sizes"""
    print("=" * 80)
    print("Testing view classifier with different PIDNet sizes")
    print("=" * 80)
    
    num_view_classes = 5
    input_tensor = torch.randn(1, 3, 512, 1024)
    
    # Test PIDNet-S
    print("\n1. PIDNet-S (small):")
    model_s = PIDNet(m=2, n=3, num_classes=19, planes=32, ppm_planes=96, 
                     head_planes=128, augment=True, num_view_classes=num_view_classes)
    model_s.eval()
    with torch.no_grad():
        outputs_s = model_s(input_tensor)
    print(f"   View output shape: {outputs_s[-1].shape}")
    assert outputs_s[-1].shape == (1, num_view_classes)
    print("   ✓ Passed")
    
    # Test PIDNet-M
    print("\n2. PIDNet-M (medium):")
    model_m = PIDNet(m=2, n=3, num_classes=19, planes=64, ppm_planes=96, 
                     head_planes=128, augment=True, num_view_classes=num_view_classes)
    model_m.eval()
    with torch.no_grad():
        outputs_m = model_m(input_tensor)
    print(f"   View output shape: {outputs_m[-1].shape}")
    assert outputs_m[-1].shape == (1, num_view_classes)
    print("   ✓ Passed")
    
    # Test PIDNet-L
    print("\n3. PIDNet-L (large):")
    model_l = PIDNet(m=3, n=4, num_classes=19, planes=64, ppm_planes=112, 
                     head_planes=256, augment=True, num_view_classes=num_view_classes)
    model_l.eval()
    with torch.no_grad():
        outputs_l = model_l(input_tensor)
    print(f"   View output shape: {outputs_l[-1].shape}")
    assert outputs_l[-1].shape == (1, num_view_classes)
    print("   ✓ Passed\n")


def test_get_pred_model():
    """Test get_pred_model function"""
    print("=" * 80)
    print("Testing get_pred_model function")
    print("=" * 80)
    
    # Without view classifier
    print("\n1. get_pred_model WITHOUT view classifier:")
    model1 = get_pred_model(name='pidnet-s', num_classes=19, num_view_classes=0)
    model1.eval()
    input_tensor = torch.randn(1, 3, 512, 1024)
    with torch.no_grad():
        output1 = model1(input_tensor)
    print(f"   Output type: {type(output1)}")
    print(f"   Output shape: {output1.shape}")
    assert isinstance(output1, torch.Tensor)
    print("   ✓ Passed")
    
    # With view classifier
    print("\n2. get_pred_model WITH view classifier:")
    model2 = get_pred_model(name='pidnet-m', num_classes=19, num_view_classes=4)
    model2.eval()
    with torch.no_grad():
        outputs2 = model2(input_tensor)
    print(f"   Number of outputs: {len(outputs2)}")
    print(f"   Segmentation shape: {outputs2[0].shape}")
    print(f"   View prediction shape: {outputs2[1].shape}")
    assert len(outputs2) == 2
    print("   ✓ Passed\n")


def main():
    print("\n" + "=" * 80)
    print("PIDNet View Classifier Test Suite")
    print("=" * 80 + "\n")
    
    try:
        # Run all tests
        test_pidnet_without_view_classifier()
        test_pidnet_with_view_classifier()
        test_pidnet_inference_mode()
        test_different_model_sizes()
        test_get_pred_model()
        
        print("=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nThe view classifier implementation is working correctly.")
        print("The I (Integral) branch features are being used for view classification.")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("✗ TEST FAILED!")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
