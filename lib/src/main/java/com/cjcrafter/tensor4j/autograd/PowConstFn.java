package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

import static jdk.incubator.vector.VectorOperators.POW;

/**
 * Backward for {@code a^p} (constant exponent {@code p}).
 *
 * <pre>{@code
 * dL/dA = p * A^(p-1) * gradOutput
 * }</pre>
 */
public class PowConstFn extends TensorFunction {

    private final float exponent;

    /**
     * Computes {@code a^p} elementwise and records the backward function.
     */
    public static Tensor apply(Tensor a, float p) {
        return applyBinaryConst(POW, t -> new PowConstFn(t, p), a, p);
    }

    /**
     * Records the backward function for {@code input^exponent}.
     */
    public PowConstFn(Tensor input, float exponent) {
        super(input);
        this.exponent = exponent;
    }

    /**
     * Computes the gradient of each input from {@code gradOutput}.
     */
    @Override
    public Tensor[] backward(Tensor gradOutput) {
        return new Tensor[]{
                inputs[0].pow(exponent - 1f).mul(exponent).mul_(gradOutput)
        };
    }
}
