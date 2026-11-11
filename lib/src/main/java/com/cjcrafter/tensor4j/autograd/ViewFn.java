package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Shape;
import com.cjcrafter.tensor4j.Tensor;

/**
 * Backward for {@link Tensor#view(int...)}. The forward only reinterprets the
 * data layout, so the grad's data is unchanged but its shape must be reshaped
 * back to match the source tensor.
 */
public class ViewFn extends TensorFunction {

    /**
     * Reshapes {@code a} to {@code newDims} (sharing storage) and records the backward function.
     */
    public static Tensor apply(Tensor a, int[] newDims) {
        Shape newShape = a.getShape().reshape(newDims);
        return Tensor.wrapResult(a.getStorage(), newShape, a.getOffset(), new ViewFn(a));
    }

    private final Shape sourceShape;

    /**
     * Creates the backward node for this operation; prefer {@code apply} over calling this directly.
     */
    public ViewFn(Tensor input) {
        super(input);
        this.sourceShape = input.getShape();
    }

    /**
     * Computes the gradient of each input from {@code gradOutput}.
     */
    @Override
    public Tensor[] backward(Tensor gradOutput) {
        return new Tensor[] {
                gradOutput.view(sourceShape.dims())
        };
    }
}
