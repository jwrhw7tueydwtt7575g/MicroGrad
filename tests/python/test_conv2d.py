"""Test Conv2d forward, backward, and numerical gradient checks."""
import numpy as np
from micrograd import tensor, nn


def test_conv2d_forward_backward():
    x_val = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ]
    w_val = [
        [
            [
                [1.0, 0.0],
                [0.0, 2.0]
            ]
        ]
    ]
    b_val = [0.5]

    x = tensor([[x_val]], requires_grad=True)  # shape (1, 1, 3, 3)
    
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0, bias=True)
    conv._params[0] = ("weight", tensor(w_val, requires_grad=True))
    conv._params[1] = ("bias", tensor(b_val, requires_grad=True))
    
    w = conv._params[0][1]
    b = conv._params[1][1]
    
    out = conv(x)
    
    expected_out = [
        [
            [
                [11.5, 14.5],
                [20.5, 23.5]
            ]
        ]
    ]
    
    out_arr = np.array(out.tolist()).reshape(out.shape())
    assert np.allclose(out_arr, expected_out), f"Expected {expected_out}, got {out_arr}"
    
    loss = out.sum()
    loss.backward()
    
    expected_dx = [
        [
            [
                [1.0, 1.0, 0.0],
                [1.0, 3.0, 2.0],
                [0.0, 2.0, 2.0]
            ]
        ]
    ]
    expected_dw = [
        [
            [
                [12.0, 16.0],
                [24.0, 28.0]
            ]
        ]
    ]
    expected_db = [4.0]
    
    dx_arr = np.array(x.grad().tolist()).reshape(x.shape())
    dw_arr = np.array(w.grad().tolist()).reshape(w.shape())
    db_arr = np.array(b.grad().tolist()).reshape(b.shape())
    
    assert np.allclose(dx_arr, expected_dx), f"Expected dx {expected_dx}, got {dx_arr}"
    assert np.allclose(dw_arr, expected_dw), f"Expected dw {expected_dw}, got {dw_arr}"
    assert np.allclose(db_arr, expected_db), f"Expected db {expected_db}, got {db_arr}"


def test_conv2d_numerical_gradcheck():
    np.random.seed(42)
    x_val = np.random.randn(1, 1, 4, 4).astype(np.float32).tolist()
    w_val = np.random.randn(1, 1, 3, 3).astype(np.float32).tolist()
    b_val = np.random.randn(1).astype(np.float32).tolist()
    
    x = tensor(x_val, requires_grad=True)
    w = tensor(w_val, requires_grad=True)
    b = tensor(b_val, requires_grad=True)
    
    stride = tensor([1.0])
    padding = tensor([1.0])
    
    out = x.conv2d(w, b, stride, padding)
    loss = out.sum()
    loss.backward()
    
    dx_analytic = x.grad().tolist()
    dw_analytic = w.grad().tolist()
    db_analytic = b.grad().tolist()
    
    eps = 1e-3
    
    # 1. Gradients for x
    x_flat = np.array(x_val).flatten().tolist()
    dx_numeric = []
    for i in range(len(x_flat)):
        old = x_flat[i]
        
        x_flat[i] = old + eps
        x_up = tensor(x_flat, shape=[1, 1, 4, 4])
        out_up = x_up.conv2d(w, b, stride, padding)
        loss_up = out_up.sum().tolist()[0]
        
        x_flat[i] = old - eps
        x_dn = tensor(x_flat, shape=[1, 1, 4, 4])
        out_dn = x_dn.conv2d(w, b, stride, padding)
        loss_dn = out_dn.sum().tolist()[0]
        
        x_flat[i] = old
        dx_numeric.append((loss_up - loss_dn) / (2 * eps))
        
    dx_analytic_arr = np.array(dx_analytic).reshape(x.shape())
    dx_numeric = np.array(dx_numeric).reshape(1, 1, 4, 4)
    assert np.allclose(dx_analytic_arr, dx_numeric, atol=1e-2)
    
    # 2. Gradients for w
    w_flat = np.array(w_val).flatten().tolist()
    dw_numeric = []
    for i in range(len(w_flat)):
        old = w_flat[i]
        
        w_flat[i] = old + eps
        w_up = tensor(w_flat, shape=[1, 1, 3, 3])
        out_up = x.conv2d(w_up, b, stride, padding)
        loss_up = out_up.sum().tolist()[0]
        
        w_flat[i] = old - eps
        w_dn = tensor(w_flat, shape=[1, 1, 3, 3])
        out_dn = x.conv2d(w_dn, b, stride, padding)
        loss_dn = out_dn.sum().tolist()[0]
        
        w_flat[i] = old
        dw_numeric.append((loss_up - loss_dn) / (2 * eps))
        
    dw_analytic_arr = np.array(dw_analytic).reshape(w.shape())
    dw_numeric = np.array(dw_numeric).reshape(1, 1, 3, 3)
    assert np.allclose(dw_analytic_arr, dw_numeric, atol=1e-2)

    # 3. Gradients for b
    b_flat = np.array(b_val).flatten().tolist()
    db_numeric = []
    for i in range(len(b_flat)):
        old = b_flat[i]
        
        b_flat[i] = old + eps
        b_up = tensor(b_flat, shape=[1])
        out_up = x.conv2d(w, b_up, stride, padding)
        loss_up = out_up.sum().tolist()[0]
        
        b_flat[i] = old - eps
        b_dn = tensor(b_flat, shape=[1])
        out_dn = x.conv2d(w, b_dn, stride, padding)
        loss_dn = out_dn.sum().tolist()[0]
        
        b_flat[i] = old
        db_numeric.append((loss_up - loss_dn) / (2 * eps))
        
    db_analytic_arr = np.array(db_analytic).reshape(b.shape())
    db_numeric = np.array(db_numeric).reshape(1)
    assert np.allclose(db_analytic_arr, db_numeric, atol=1e-2)
