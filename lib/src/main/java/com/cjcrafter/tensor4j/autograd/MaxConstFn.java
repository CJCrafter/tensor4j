package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

/**
 * Backward for {@code ReLU: max(x, b)}.
 *
 * <pre>{@code
 * dL/dInput = gradOutput where input > b, else 0
 * }</pre>
 *
 * <p>Equivalent to: {@code gradOutput * (input > b)}.
 */
public class MaxConstFn extends TensorFunction {

    private final float b;

    public MaxConstFn(Tensor input, float b) {
        super(input);
        this.b = b;
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        Tensor input = inputs[0];
        // mask: 1 where input > 0, 0 elsewhere
        Tensor mask = input.gt(b);
        return new Tensor[]{
                mask.mul_(gradOutput)
        };
    }
}
