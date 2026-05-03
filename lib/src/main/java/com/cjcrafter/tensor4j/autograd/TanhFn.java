package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

import static jdk.incubator.vector.VectorOperators.TANH;

/**
 * Backward for {@code tanh(a)}.
 *
 * <pre>{@code
 *  dL/dInput = (1 - tanh(input)^2) * gradOutput
 * }</pre>
 */
public class TanhFn extends TensorFunction {

    public static Tensor apply(Tensor a) {
        return applyUnaryCached(TANH, TanhFn::new, a);
    }

    private final Tensor result;

    public TanhFn(Tensor input, Tensor result) {
        super(input);
        this.result = result;
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        Tensor tanh2 = result.square().rsub_(1);
        return new Tensor[]{ tanh2.mul_(gradOutput) };
    }
}
