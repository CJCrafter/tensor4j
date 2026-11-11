package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

import static jdk.incubator.vector.VectorOperators.EXP;

/**
 * Backward for {@code exp(A)}.
 *
 * <pre>{@code
 * dL/dA = exp(A) * gradOutput
 * }</pre>
 */
public class ExpFn extends TensorFunction {

    /**
     * Computes {@code exp(a)} elementwise and records the backward function.
     */
    public static Tensor apply(Tensor a) {
        return applyUnaryCached(EXP, ExpFn::new, a);
    }

    // cache this to avoid recomputations
    private final Tensor result;

    /**
     * Creates the backward node for this operation; prefer {@code apply} over calling this directly.
     */
    public ExpFn(Tensor a, Tensor result) {
        super(a);
        this.result = result;
    }

    /**
     * Computes the gradient of each input from {@code gradOutput}.
     */
    @Override
    public Tensor[] backward(Tensor gradOutput) {
        return new Tensor[] {
                result.mul(gradOutput)
        };
    }
}
