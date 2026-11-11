package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

import static jdk.incubator.vector.VectorOperators.ADD;

/**
 * Backward for {@code C = A + B}.
 *
 * <pre>{@code
 * dL/dA = gradOutput
 * dL/dB = gradOutput
 * }</pre>
 */
public class AddFn extends TensorFunction {

    /**
     * Computes {@code a + b} (broadcasting shapes as needed) and records the backward function.
     */
    public static Tensor apply(Tensor a, Tensor b) {
        return applyBinary(ADD, AddFn::new, a, b);
    }

    /**
     * Creates the backward node for this operation; prefer {@code apply} over calling this directly.
     */
    public AddFn(Tensor a, Tensor b) {
        super(a, b);
    }

    /**
     * Computes the gradient of each input from {@code gradOutput}.
     */
    @Override
    public Tensor[] backward(Tensor gradOutput) {
        return new Tensor[]{
            unbroadcast(gradOutput, inputs[0].getShape()),
            unbroadcast(gradOutput, inputs[1].getShape())
        };
    }
}
