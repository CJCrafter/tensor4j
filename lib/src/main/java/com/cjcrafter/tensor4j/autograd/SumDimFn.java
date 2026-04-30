package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;

/**
 * Backward for {@code sum(dim)}.
 */
public class SumDimFn extends TensorFunction {

    private final int dim;

    public SumDimFn(Tensor input, int dim) {
        super(input);
        this.dim = dim;
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        // gradOutput has shape with dim collapsed to 1.
        // Broadcasting naturally repeats it along that dimension
        // to match the original input shape.
        Tensor ones = TensorBuilder.builder()
                .shape(inputs[0].getShape())
                .ones();
        return new Tensor[]{ ones.mul(gradOutput) };
    }
}
