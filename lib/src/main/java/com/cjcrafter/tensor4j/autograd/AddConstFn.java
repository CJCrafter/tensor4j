package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

/**
 * Backward for {@code C = A + b} (or {@code C = A - b}).
 *
 * <pre>{@code
 * dL/dA = gradOutput
 * }</pre>
 */
public class AddConstFn extends TensorFunction {

    public AddConstFn(Tensor a, float b) {
        super(a);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        return new Tensor[]{
                gradOutput.clone()
        };
    }
}
