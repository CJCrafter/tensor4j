package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

/**
 * Backward for {@link Tensor#transpose()}. The forward swaps the last two
 * dimensions, so the grad must be transposed back so its shape matches the
 * source tensor.
 */
public class TransposeFn extends TensorFunction {

    /**
     * Swaps the last two dimensions of {@code a} (sharing storage) and records the backward function.
     */
    public static Tensor apply(Tensor a) {
        return Tensor.wrapResult(a.getStorage(), a.getShape().transpose(),
                a.getOffset(), new TransposeFn(a));
    }

    /**
     * Creates the backward node for this operation; prefer {@code apply} over calling this directly.
     */
    public TransposeFn(Tensor input) {
        super(input);
    }

    /**
     * Computes the gradient of each input from {@code gradOutput}.
     */
    @Override
    public Tensor[] backward(Tensor gradOutput) {
        return new Tensor[] {
                gradOutput.transpose().contiguous()
        };
    }
}
