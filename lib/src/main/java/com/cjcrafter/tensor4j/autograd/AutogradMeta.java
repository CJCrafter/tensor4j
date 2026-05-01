package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

public final class AutogradMeta {

    public Tensor grad;
    public TensorFunction gradFn;
}
