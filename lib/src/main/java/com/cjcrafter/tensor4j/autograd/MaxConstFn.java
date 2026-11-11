package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

import static jdk.incubator.vector.VectorOperators.MAX;

/**
 * Backward for {@code max(x, b)} (constant {@code b}; e.g. ReLU is the special case {@code b = 0}).
 *
 * <pre>{@code
 * dL/dInput = gradOutput where input > b, else 0
 * }</pre>
 *
 * <p>Equivalent to: {@code gradOutput * (input > b)}.
 */
public class MaxConstFn extends TensorFunction {

    /**
     * Computes {@code max(a, b)} and records the backward function.
     */
    public static Tensor apply(Tensor a, float b) {
        return applyBinaryConst(MAX, t -> new MaxConstFn(t, b), a, b);
    }

    private final float b;

    /**
     * Creates the backward node for this operation; prefer {@code apply} over calling this directly.
     */
    public MaxConstFn(Tensor input, float b) {
        super(input);
        this.b = b;
    }

    /**
     * Computes the gradient of each input from {@code gradOutput}.
     */
    @Override
    public Tensor[] backward(Tensor gradOutput) {
        Tensor input = inputs[0];
        // mask: 1 where input > b, 0 elsewhere
        Tensor mask = input.gt(b);
        return new Tensor[]{
                mask.mul_(gradOutput)
        };
    }
}
