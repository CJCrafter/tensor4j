package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

import static jdk.incubator.vector.VectorOperators.ADD;
import static jdk.incubator.vector.VectorOperators.SUB;

/**
 * Backward for {@code C = A + b} (or {@code C = A - b}).
 *
 * <pre>{@code
 * dL/dA = gradOutput
 * }</pre>
 */
public class AddConstFn extends TensorFunction {

    /**
     * Computes {@code a + b} and records the backward function.
     */
    public static Tensor applyAdd(Tensor a, float b) {
        return applyBinaryConst(ADD, t -> new AddConstFn(t, b), a, b);
    }

    /**
     * Computes {@code a - b} and records the backward function.
     */
    public static Tensor applySub(Tensor a, float b) {
        return applyBinaryConst(SUB, t -> new AddConstFn(t, b), a, b);
    }

    /**
     * Creates the backward node for this operation; prefer {@code apply} over calling this directly.
     */
    public AddConstFn(Tensor a, float b) {
        super(a);
    }

    /**
     * Computes the gradient of each input from {@code gradOutput}.
     */
    @Override
    public Tensor[] backward(Tensor gradOutput) {
        return new Tensor[]{
                gradOutput.clone()
        };
    }
}
