package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

/**
 * Backward for {@code max(x, b)}.
 *
 * <pre>{@code
 * dL/dInput = gradOutput where input > b, else 0
 * }</pre>
 */
public class MaxFn extends TensorFunction {

    public MaxFn(Tensor a, Tensor b) {
        super(a, b);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        // mask: 1 where a >= b, 0 elsewhere
        Tensor maskA = inputs[0].ge(inputs[1]);
        Tensor maskB = inputs[0].lt(inputs[1]);
        return new Tensor[]{
            unbroadcast(maskA.mul_(gradOutput), inputs[0].getShape()),
            unbroadcast(maskB.mul_(gradOutput), inputs[1].getShape())
        };
    }
}