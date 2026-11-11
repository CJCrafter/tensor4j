package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

import static jdk.incubator.vector.VectorOperators.MUL;

/**
 * Backward for {@code C = A * b} (element-wise).
 *
 * <pre>{@code
 * dL/dA = gradOutput * b
 * }</pre>
 */
public class MulConstFn extends TensorFunction {

    /**
     * Computes {@code a * b} and records the backward function.
     */
    public static Tensor apply(Tensor a, float b) {
        return applyBinaryConst(MUL, t -> new MulConstFn(t, b), a, b);
    }

    private final float b;

    /**
     * Creates the backward node for this operation; prefer {@code apply} over calling this directly.
     */
    public MulConstFn(Tensor a, float b) {
        super(a);
        this.b = b;
    }

    /**
     * Computes the gradient of each input from {@code gradOutput}.
     */
    @Override
    public Tensor[] backward(Tensor gradOutput) {
        return new Tensor[]{
                gradOutput.mul(b),
        };
    }
}