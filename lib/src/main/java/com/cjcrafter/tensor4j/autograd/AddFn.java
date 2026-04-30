package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

/**
 * Backward for {@code C = A + B}.
 *
 * <pre>{@code
 * dL/dA = gradOutput
 * dL/dB = gradOutput
 * }</pre>
 */
public class AddFn extends TensorFunction {

    public AddFn(Tensor a, Tensor b) {
        super(a, b);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        return new Tensor[]{
            unbroadcast(gradOutput, inputs[0].getShape()),
            unbroadcast(gradOutput, inputs[1].getShape())
        };
    }
}
