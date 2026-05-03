package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

import static jdk.incubator.vector.VectorOperators.SUB;

/**
 * Backward for {@code C = A - B}.
 *
 * <pre>{@code
 * dL/dA = gradOutput
 * dL/dB = -gradOutput
 * }</pre>
 */
public class SubFn extends TensorFunction {

    public static Tensor apply(Tensor a, Tensor b) {
        return applyBinary(SUB, SubFn::new, a, b);
    }

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
