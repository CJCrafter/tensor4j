package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Shape;
import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.memory.Storage;
import com.cjcrafter.tensor4j.ops.Ops;
import com.google.common.base.Preconditions;

/**
 * Backward for {@code max(dim)}.
 *
 * <pre>{@code
 *     dL/dx[i] = gradOutput[reduced(i)] * 1[x[i] == y[reduced(i)]] / ties
 * }</pre>
 */
public class MaxDimFn extends TensorFunction {

    public static Tensor apply(Tensor a, int dim) {
        Preconditions.checkArgument(dim >= 0 && dim < a.getShape().dimensions(),
                "dim %s out of range for %sD tensor", dim, a.getShape().dimensions());
        Storage out = Ops.maxDim(a, dim);
        int[] newDims = a.getShape().dims();
        newDims[dim] = 1;
        Shape resultShape = new Shape(newDims);
        return Tensor.wrapResultDeferred(out, resultShape, new Tensor[]{a},
                cached -> new MaxDimFn(a, cached, dim));
    }

    private final Tensor result;
    private final int dim;

    public MaxDimFn(Tensor input, Tensor result, int dim) {
        super(input);
        this.result = result;
        this.dim = dim;
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        Tensor input = inputs[0];

        // mask = 1 where input == max along dim (broadcast), else 0
        Tensor mask = input.eq(result);

        // count ties per reduced slice, then divide the mask by the count so
        // that grad gets split evenly among tied maxima.
        Tensor invTies = mask.sum(dim).rdiv_(1f);
        return new Tensor[]{
                mask.mul_(invTies).mul_(gradOutput)
        };
    }
}
