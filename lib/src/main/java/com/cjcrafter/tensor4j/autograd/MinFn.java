package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

/**
 * Backward for elementwise {@code min(a, b)}.
 *
 * <pre>{@code
 * dL/dA = gradOutput where a <= b, else 0
 * dL/dB = gradOutput where a >  b, else 0
 * }</pre>
 *
 * <p>Mirrors {@link MaxFn}. Ties go entirely to {@code a} so gradient
 * magnitude is preserved.
 */
public class MinFn extends TensorFunction {

    public MinFn(Tensor a, Tensor b) {
        super(a, b);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        Tensor maskA = inputs[0].le(inputs[1]);
        Tensor maskB = inputs[0].gt(inputs[1]);
        return new Tensor[]{
            unbroadcast(maskA.mul_(gradOutput), inputs[0].getShape()),
            unbroadcast(maskB.mul_(gradOutput), inputs[1].getShape())
        };
    }
}
