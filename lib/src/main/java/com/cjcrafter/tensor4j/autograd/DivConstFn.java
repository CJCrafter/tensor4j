package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

import static jdk.incubator.vector.VectorOperators.DIV;

/**
 * Backward for {@code C = A / b} (scalar divisor).
 *
 * <pre>{@code
 * dL/dA = gradOutput * (1 / b)
 * }</pre>
 */
public class DivConstFn extends TensorFunction {

    /**
     * Computes {@code a / b} and records the backward function.
     */
    public static Tensor apply(Tensor a, float b) {
        return applyBinaryConst(DIV, t -> new DivConstFn(t, b), a, b);
    }

    private final float b;

    /**
     * Creates the backward node for this operation; prefer {@code apply} over calling this directly.
     */
    public DivConstFn(Tensor a, float b) {
        super(a);
        this.b = b;
    }

    /**
     * Computes the gradient of each input from {@code gradOutput}.
     */
    @Override
    public Tensor[] backward(Tensor gradOutput) {
        return new Tensor[]{
                gradOutput.mul(1.0f / b),
        };
    }
}
