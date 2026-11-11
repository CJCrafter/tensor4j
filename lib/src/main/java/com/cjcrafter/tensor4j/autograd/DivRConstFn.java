package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

import static jdk.incubator.vector.VectorOperators.DIV;

/**
 * Backward for {@code C = a / B} (scalar dividend).
 *
 * <pre>{@code
 * dL/dB = -a / B^2 * gradOutput
 * }</pre>
 */
public class DivRConstFn extends TensorFunction {

    /**
     * Computes {@code a / b} and records the backward function.
     */
    public static Tensor apply(float a, Tensor b) {
        return applyRBinaryConst(DIV, t -> new DivRConstFn(a, t), a, b);
    }

    private final float a;

    /**
     * Creates the backward node for this operation; prefer {@code apply} over calling this directly.
     */
    public DivRConstFn(float a, Tensor b) {
        super(b);
        this.a = a;
    }

    /**
     * Computes the gradient of each input from {@code gradOutput}.
     */
    @Override
    public Tensor[] backward(Tensor gradOutput) {
        return new Tensor[]{
                inputs[0].square().rdiv_(-a).mul_(gradOutput),
        };
    }
}
