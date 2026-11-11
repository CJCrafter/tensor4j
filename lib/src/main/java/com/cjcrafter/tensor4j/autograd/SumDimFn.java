package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Shape;
import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;
import com.cjcrafter.tensor4j.memory.Storage;
import com.cjcrafter.tensor4j.ops.Ops;
import com.google.common.base.Preconditions;

/**
 * Backward for {@code sum(dim)}.
 */
public class SumDimFn extends TensorFunction {

    /**
     * Sums {@code a} along {@code dim} (collapsed to size 1) and records the backward function.
     */
    public static Tensor apply(Tensor a, int dim) {
        Preconditions.checkArgument(dim >= 0 && dim < a.getShape().dimensions(),
                "dim %s out of range for %sD tensor", dim, a.getShape().dimensions());
        Storage out = Ops.sumDim(a, dim);
        int[] newDims = a.getShape().dims();
        newDims[dim] = 1;
        return Tensor.wrapResult(out, new Shape(newDims), new SumDimFn(a, dim));
    }

    private final int dim;

    /**
     * Creates the backward node for this operation; prefer {@code apply} over calling this directly.
     */
    public SumDimFn(Tensor input, int dim) {
        super(input);
        this.dim = dim;
    }

    /**
     * Computes the gradient of each input from {@code gradOutput}.
     */
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
