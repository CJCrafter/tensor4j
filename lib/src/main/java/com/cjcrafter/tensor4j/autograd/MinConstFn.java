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

    public static Tensor apply(Tensor a, float b) {
        return applyBinaryConst(MIN, t -> new MinConstFn(t, b), a, b);
    }

    private final float b;

    public MinConstFn(Tensor input, float b) {
        super(input);
        this.b = b;
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        Tensor input = inputs[0];
        Tensor mask = input.lt(b);
        return new Tensor[]{ mask.mul_(gradOutput) };
    }
}
