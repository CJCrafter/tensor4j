package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

import static jdk.incubator.vector.VectorOperators.MUL;

/**
 * Backward for {@code C = A * B} (element-wise).
 *
 * <pre>{@code
 * dL/dA = gradOutput * B
 * dL/dB = gradOutput * A
 * }</pre>
 */
public class MulFn extends TensorFunction {

    public static Tensor apply(Tensor a, Tensor b) {
        return applyBinary(MUL, MulFn::new, a, b);
    }

    public MulFn(Tensor a, Tensor b) {
        super(a, b);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        return new Tensor[]{
            unbroadcast(gradOutput.mul(inputs[1]), inputs[0].getShape()),
            unbroadcast(gradOutput.mul(inputs[0]), inputs[1].getShape())
        };
    }
}
