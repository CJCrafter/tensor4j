package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

import static jdk.incubator.vector.VectorOperators.MIN;

/**
 * Backward for {@code min(x, b)} with constant {@code b}.
 *
 * <pre>{@code
 * dL/dInput = gradOutput where input < b, else 0
 * }</pre>
 *
 * <p>Mirrors {@link MaxConstFn}.
 */
public class MinConstFn extends TensorFunction {

    /**
     * Computes {@code min(a, b)} and records the backward function.
     */
    public static Tensor apply(Tensor a, float b) {
        return applyBinaryConst(MIN, t -> new MinConstFn(t, b), a, b);
    }

    private final float b;

    /**
     * Creates the backward node for this operation; prefer {@code apply} over calling this directly.
     */
    public MinConstFn(Tensor input, float b) {
        super(input);
        this.b = b;
    }

    /**
     * Computes the gradient of each input from {@code gradOutput}.
     */
    @Override
    public Tensor[] backward(Tensor gradOutput) {
        Tensor input = inputs[0];
        Tensor mask = input.lt(b);
        return new Tensor[]{ mask.mul_(gradOutput) };
    }
}
