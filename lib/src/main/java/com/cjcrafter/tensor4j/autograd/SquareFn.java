package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

/**
 * Backward for {@code C = A ^ 2}.
 *
 * <pre>{@code
 * dL/dA = 2 * A * gradOutput
 * }</pre>
 */
public class SquareFn extends TensorFunction {

    public SquareFn(Tensor a) {
        super(a);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        return new Tensor[]{
                gradOutput.mul(2).mul_(inputs[0]),
        };
    }
}