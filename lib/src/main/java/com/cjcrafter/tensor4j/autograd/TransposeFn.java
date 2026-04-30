package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

/**
 * Backward for {@link Tensor#transpose()}. The forward swaps the last two
 * dimensions, so the grad must be transposed back so its shape matches the
 * source tensor.
 */
public class TransposeFn extends TensorFunction {

    public TransposeFn(Tensor input) {
        super(input);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        return new Tensor[] {
                gradOutput.transpose().contiguous()
        };
    }
}
