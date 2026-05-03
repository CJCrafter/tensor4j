package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

import static jdk.incubator.vector.VectorOperators.SUB;

/**
 * Backward for {@code C = a - B}.
 *
 * <pre>{@code
 * dL/dB = -gradOutput
 * }</pre>
 */
public class SubRConstFn extends TensorFunction {

    public static Tensor apply(float a, Tensor b) {
        return applyRBinaryConst(SUB, t -> new SubRConstFn(a, t), a, b);
    }

    public SubRConstFn(float a, Tensor b) {
        super(b);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        Tensor negGrad = gradOutput.mul(-1);
        return new Tensor[]{
                unbroadcast(negGrad, inputs[0].getShape())
        };
    }
}
