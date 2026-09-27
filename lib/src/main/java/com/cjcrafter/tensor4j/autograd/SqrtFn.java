package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

import static jdk.incubator.vector.VectorOperators.SQRT;

/**
 * Backward for {@code sqrt(A)}.
 *
 * <pre>{@code
 * dL/dA = (1 / 2) / sqrt(A) * gradOutput
 * }</pre>
 */
public class SqrtFn extends TensorFunction {

    /**
     * Computes {@code sqrt(a)} elementwise and records the backward function.
     */
    public static Tensor apply(Tensor a) {
        return applyUnaryCached(SQRT, SqrtFn::new, a);
    }

    private final Tensor result;

    public SqrtFn(Tensor a, Tensor result) {
        super(a);
        this.result = result;
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        return new Tensor[] {
                result.rdiv(0.5f).mul_(gradOutput)
        };
    }
}
