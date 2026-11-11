package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

/**
 * Backward for ops that copy data without changing its values (e.g. {@link Tensor#clone()},
 * {@link Tensor#contiguous()}); passes the gradient straight through unchanged.
 */
public class IdentityFn extends TensorFunction {

    /**
     * Records the backward function for a data copy of {@code input}.
     */
    public IdentityFn(Tensor input) {
        super(input);
    }

    /**
     * Computes the gradient of each input from {@code gradOutput}.
     */
    @Override
    public Tensor[] backward(Tensor gradOutput) {
        return new Tensor[] {
                gradOutput.clone()
        };
    }
}
