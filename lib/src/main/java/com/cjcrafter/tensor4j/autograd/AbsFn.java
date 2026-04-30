package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

/**
 * Backward for {@code abs(a)}.
 *
 * <pre>{@code
 * dL/dA = sign(A) * gradOutput
 * }</pre>
 */
public class AbsFn extends TensorFunction {

    public AbsFn(Tensor a) {
        super(a);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        // sign(x): 1 for x > 0, -1 for x < 0, 0 for x == 0
        Tensor pos = inputs[0].gt(0f);
        Tensor neg = inputs[0].lt(0f).mul_(-1f);
        Tensor sign = pos.add_(neg);
        return new Tensor[]{ sign.mul_(gradOutput) };
    }
}
