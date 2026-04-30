package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

/**
 * Backward for {@code log(A)} (natural log).
 *
 * <pre>{@code
 * dL/dA = (1 / A) * gradOutput
 * }</pre>
 */
public class LogFn extends TensorFunction {

    public LogFn(Tensor a) {
        super(a);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        return new Tensor[] {
                inputs[0].rdiv(1.0f).mul_(gradOutput)
        };
    }
}
