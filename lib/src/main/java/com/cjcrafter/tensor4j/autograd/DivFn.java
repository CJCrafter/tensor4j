package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

import static jdk.incubator.vector.VectorOperators.DIV;

/**
 * Backward for {@code A / B}.
 *
 * <pre>{@code
 * dL/dA = 1/B * gradOutput
 * dL/dB = -A/B^2 * gradOutput
 * }</pre>
 */
public class DivFn extends TensorFunction {

    /**
     * Computes {@code a / b} elementwise (broadcasting shapes as needed) and records the
     * backward function.
     */
    public static Tensor apply(Tensor a, Tensor b) {
        return applyBinary(DIV, DivFn::new, a, b);
    }

    /**
     * Creates the backward node for this operation; prefer {@code apply} over calling this directly.
     */
    public DivFn(Tensor a, Tensor b) {
        super(a, b);
    }

    /**
     * Computes the gradient of each input from {@code gradOutput}.
     */
    @Override
    public Tensor[] backward(Tensor gradOutput) {
        Tensor gradA = gradOutput.div(inputs[1]);
        Tensor gradB = gradOutput.mul(inputs[0]).mul(-1f).div(inputs[1].square());
        return new Tensor[]{
                unbroadcast(gradA, inputs[0].getShape()),
                unbroadcast(gradB, inputs[1].getShape())
        };
    }
}
