package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;

/**
 * Backward for {@code mean} (full reduction to scalar).
 *
 * <pre>{@code
 * dL/dInput = gradOutput / numel, broadcast to input's shape
 * }</pre>
 */
public class MeanFn extends TensorFunction {

    public MeanFn(Tensor input) {
        super(input);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        int n = inputs[0].getShape().numel();
        Tensor scale = TensorBuilder.builder()
                .shape(1, 1)
                .fill(1f / n);
        Tensor scaled = gradOutput.mul(scale);

        Tensor ones = TensorBuilder.builder()
                .shape(inputs[0].getShape())
                .ones();
        return new Tensor[]{ ones.mul(scaled) };
    }
}
