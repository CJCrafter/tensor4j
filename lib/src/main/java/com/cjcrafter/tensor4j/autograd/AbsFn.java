package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

import static jdk.incubator.vector.VectorOperators.ABS;

/**
 * Backward for {@code abs(a)}.
 *
 * <pre>{@code
 * dL/dA = sign(A) * gradOutput
 * }</pre>
 */
public class AbsFn extends TensorFunction {

    /**
     * Computes {@code abs(a)} elementwise and records the backward function.
     */
    public static Tensor apply(Tensor a) {
        return applyUnary(ABS, AbsFn::new, a);
    }

    /**
     * Creates the backward node for this operation; prefer {@code apply} over calling this directly.
     */
    public AbsFn(Tensor a) {
        super(a);
    }

    /**
     * Computes the gradient of each input from {@code gradOutput}.
     */
    @Override
    public Tensor[] backward(Tensor gradOutput) {
        // sign(x): 1 for x > 0, -1 for x < 0, 0 for x == 0
        Tensor pos = inputs[0].gt(0f);
        Tensor neg = inputs[0].lt(0f).mul_(-1f);
        Tensor sign = pos.add_(neg);
        return new Tensor[]{ sign.mul_(gradOutput) };
    }
}
