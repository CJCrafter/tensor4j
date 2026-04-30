package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Shape;
import com.cjcrafter.tensor4j.Tensor;

/**
 * Backward for {@link Tensor#view(int...)}. The forward only reinterprets the
 * data layout, so the grad's data is unchanged but its shape must be reshaped
 * back to match the source tensor.
 */
public class ViewFn extends TensorFunction {

    private final Shape sourceShape;

    public ViewFn(Tensor input) {
        super(input);
        this.sourceShape = input.getShape();
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        return new Tensor[] {
                gradOutput.view(sourceShape.dims())
        };
    }
}
