package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

/**
 * Backward for {@code max(dim)}.
 *
 * <pre>{@code
 *     dL/dx[i] = gradOutput[reduced(i)] * 1[x[i] == y[reduced(i)]] / ties
 * }</pre>
 */
public class MaxDimFn extends TensorFunction {

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
