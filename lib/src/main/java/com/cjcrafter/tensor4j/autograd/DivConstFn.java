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

    public DivConstFn(Tensor a, float b) {
        super(a);
        this.b = b;
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        return new Tensor[]{
                gradOutput.mul(1.0f / b),
        };
    }
}
