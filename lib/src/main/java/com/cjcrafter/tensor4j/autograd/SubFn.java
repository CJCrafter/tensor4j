package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;

/**
 * Backward for {@code C = A - B}.
 *
 * <pre>{@code
 * dL/dA = gradOutput
 * dL/dB = -gradOutput
 * }</pre>
 */
public class SubFn extends TensorFunction {

    public SubFn(Tensor a, Tensor b) {
        super(a, b);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        Tensor negGrad = gradOutput.mul(-1);
        return new Tensor[]{
            unbroadcast(gradOutput, inputs[0].getShape()),
            unbroadcast(negGrad, inputs[1].getShape())
        };
    }
}
