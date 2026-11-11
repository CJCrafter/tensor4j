package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

import static jdk.incubator.vector.VectorOperators.MUL;

/**
 * Backward for {@code C = A * B} (element-wise).
 *
 * <pre>{@code
 * dL/dA = gradOutput * B
 * dL/dB = gradOutput * A
 * }</pre>
 */
public class MulFn extends TensorFunction {

    /**
     * Computes {@code a * b} elementwise (broadcasting shapes as needed) and records the
     * backward function.
     */
    public static Tensor apply(Tensor a, Tensor b) {
        return applyBinary(MUL, MulFn::new, a, b);
    }

    /**
     * Creates the backward node for this operation; prefer {@code apply} over calling this directly.
     */
    public MulFn(Tensor a, Tensor b) {
        super(a, b);
    }

    /**
     * Computes the gradient of each input from {@code gradOutput}.
     */
    @Override
    public Tensor[] backward(Tensor gradOutput) {
        return new Tensor[]{
            unbroadcast(gradOutput.mul(inputs[1]), inputs[0].getShape()),
            unbroadcast(gradOutput.mul(inputs[0]), inputs[1].getShape())
        };
    }
}
