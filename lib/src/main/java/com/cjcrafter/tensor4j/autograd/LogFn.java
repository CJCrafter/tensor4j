package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

import static jdk.incubator.vector.VectorOperators.LOG;

/**
 * Backward for {@code log(A)} (natural log).
 *
 * <pre>{@code
 * dL/dA = (1 / A) * gradOutput
 * }</pre>
 */
public class LogFn extends TensorFunction {

    /**
     * Computes {@code log(a)} elementwise and records the backward function.
     */
    public static Tensor apply(Tensor a) {
        return applyUnary(LOG, LogFn::new, a);
    }

    /**
     * Creates the backward node for this operation; prefer {@code apply} over calling this directly.
     */
    public LogFn(Tensor a) {
        super(a);
    }

    /**
     * Computes the gradient of each input from {@code gradOutput}.
     */
    @Override
    public Tensor[] backward(Tensor gradOutput) {
        return new Tensor[] {
                inputs[0].rdiv(1.0f).mul_(gradOutput)
        };
    }
}
