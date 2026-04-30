package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;

/**
 * Backward for {@code sum} (full reduction to scalar).
 *
 * <pre>{@code
 * dL/dInput = gradOutput broadcast to input's shape
 * }</pre>
 */
public class SumFn extends TensorFunction {

    public SumFn(Tensor input) {
        super(input);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        // gradOutput is scalar (1,) broadcast to input shape
        // dL/dInput[i] = dL/dSum = gradOutput for every element
        Tensor ones = TensorBuilder.builder()
                .shape(inputs[0].getShape())
                .ones();
        Tensor expanded = ones.mul(gradOutput);
        return new Tensor[]{ expanded };
    }
}
