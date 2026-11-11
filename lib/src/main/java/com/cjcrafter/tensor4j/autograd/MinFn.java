package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

import static jdk.incubator.vector.VectorOperators.MIN;

/**
 * Backward for elementwise {@code min(a, b)}.
 *
 * <pre>{@code
 * dL/dA = gradOutput where a <= b, else 0
 * dL/dB = gradOutput where a >  b, else 0
 * }</pre>
 *
 * <p>Mirrors {@link MaxFn}. Ties go entirely to {@code a} so gradient
 * magnitude is preserved.
 */
public class MinFn extends TensorFunction {

    /**
     * Computes the elementwise minimum of {@code a} and {@code b} (broadcasting shapes as
     * needed) and records the backward function.
     */
    public static Tensor apply(Tensor a, Tensor b) {
        return applyBinary(MIN, MinFn::new, a, b);
    }

    /**
     * Creates the backward node for this operation; prefer {@code apply} over calling this directly.
     */
    public MinFn(Tensor a, Tensor b) {
        super(a, b);
    }

    /**
     * Computes the gradient of each input from {@code gradOutput}.
     */
    @Override
    public Tensor[] backward(Tensor gradOutput) {
        Tensor maskA = inputs[0].le(inputs[1]);
        Tensor maskB = inputs[0].gt(inputs[1]);
        return new Tensor[]{
            unbroadcast(maskA.mul_(gradOutput), inputs[0].getShape()),
            unbroadcast(maskB.mul_(gradOutput), inputs[1].getShape())
        };
    }
}
