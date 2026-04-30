package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

/**
 * Backward for {@code C = A / b} (scalar divisor).
 *
 * <pre>{@code
 * dL/dA = gradOutput * (1 / b)
 * }</pre>
 */
public class DivConstFn extends TensorFunction {

    private final float b;

    public DivConstFn(Tensor a, float b) {
        super(a);
        this.b = b;
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        return new Tensor[]{
                gradOutput.mul(1.0f / b),
        };
    }
}
