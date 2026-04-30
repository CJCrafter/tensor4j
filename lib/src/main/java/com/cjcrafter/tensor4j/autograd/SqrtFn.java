package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

/**
 * Backward for {@code sqrt(A)}.
 *
 * <pre>{@code
 * dL/dA = (1 / 2) / sqrt(A) * gradOutput
 * }</pre>
 */
public class SqrtFn extends TensorFunction {

    private final Tensor result;

    public SqrtFn(Tensor a, Tensor result) {
        super(a);
        this.result = result;
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        return new Tensor[] {
                result.rdiv(0.5f).mul_(gradOutput)
        };
    }
}
