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

    public static Tensor apply(Tensor a) {
        return applyUnaryCached(EXP, ExpFn::new, a);
    }

    // cache this to avoid recomputations
    private final Tensor result;

    public ExpFn(Tensor a, Tensor result) {
        super(a);
        this.result = result;
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        return new Tensor[] {
                result.mul(gradOutput)
        };
    }
}
