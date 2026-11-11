package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.memory.Storage;
import com.cjcrafter.tensor4j.ops.Ops;

import static jdk.incubator.vector.VectorOperators.MUL;

/**
 * Backward for {@code C = A ^ 2}.
 *
 * <pre>{@code
 * dL/dA = 2 * A * gradOutput
 * }</pre>
 */
public class SquareFn extends TensorFunction {

    /**
     * Computes {@code a^2} elementwise and records the backward function.
     */
    public static Tensor apply(Tensor a) {
        Storage out = Ops.binaryOp(MUL, a, a, a.getShape());
        return Tensor.wrapResult(out, a.getShape(), new SquareFn(a));
    }

    /**
     * Creates the backward node for this operation; prefer {@code apply} over calling this directly.
     */
    public SquareFn(Tensor a) {
        super(a);
    }

    /**
     * Computes the gradient of each input from {@code gradOutput}.
     */
    @Override
    public Tensor[] backward(Tensor gradOutput) {
        return new Tensor[]{
                gradOutput.mul(2).mul_(inputs[0]),
        };
    }
}