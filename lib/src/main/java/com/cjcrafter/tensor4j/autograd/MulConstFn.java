package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

/**
 * Backward for {@code C = A * b} (element-wise).
 *
 * <pre>{@code
 * dL/dA = gradOutput * b
 * }</pre>
 */
public class MulConstFn extends TensorFunction {

    private final float b;

    public MulConstFn(Tensor a, float b) {
        super(a);
        this.b = b;
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        return new Tensor[]{
                gradOutput.mul(b),
        };
    }
}