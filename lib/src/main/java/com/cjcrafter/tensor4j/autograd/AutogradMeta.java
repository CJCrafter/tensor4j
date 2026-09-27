package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

/**
 * Lazily-allocated per-tensor autograd state: the accumulated gradient and the function
 * that produced this tensor (if any).
 */
public final class AutogradMeta {

    /** The gradient accumulated so far from {@link Tensor#backward()}; {@code null} until first accumulated. */
    public Tensor grad;

    /** The function that produced this tensor, or {@code null} if this tensor is a leaf. */
    public TensorFunction gradFn;
}
