package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

/**
 * Backward for {@code C = A @ B}.
 *
 * <pre>{@code
 * dL/dA = gradOutput @ B^T
 * dL/dB = A^T @ gradOutput
 * }</pre>
 */
public class MatmulFn extends TensorFunction {

    public MatmulFn(Tensor a, Tensor b) {
        super(a, b);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        Tensor a = inputs[0];
        Tensor b = inputs[1];
        Tensor gradA = unbroadcast(gradOutput.matmul(b.transpose()), a.getShape());
        Tensor gradB = unbroadcast(a.transpose().matmul(gradOutput), b.getShape());
        return new Tensor[]{gradA, gradB};
    }
}
