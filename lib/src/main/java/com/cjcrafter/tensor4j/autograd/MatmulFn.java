package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Shape;
import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.memory.Storage;
import com.cjcrafter.tensor4j.ops.Ops;
import com.google.common.base.Preconditions;

/**
 * Backward for {@code C = A @ B}.
 *
 * <pre>{@code
 * dL/dA = gradOutput @ B^T
 * dL/dB = A^T @ gradOutput
 * }</pre>
 */
public class MatmulFn extends TensorFunction {

    public static Tensor apply(Tensor a, Tensor b) {
        int aDim = a.getShape().dimensions();
        int bDim = b.getShape().dimensions();

        Preconditions.checkArgument(aDim <= 3 && bDim <= 3,
                "matmul requires 2D or 3D tensors, got %sD and %sD", aDim, bDim);

        int k1 = a.getShape().dim(aDim - 1);
        int k2 = b.getShape().dim(bDim - 2);
        Preconditions.checkArgument(k1 == k2,
                "matmul inner dimensions don't match: %s @ %s", a.getShape(), b.getShape());

        int m = a.getShape().dim(aDim - 2);
        int n = b.getShape().dim(bDim - 1);
        Shape resultShape;

        if (aDim == 2 && bDim == 2) {
            resultShape = new Shape(m, n);
        } else {
            int batch;
            if (aDim == 3 && bDim == 3) {
                Preconditions.checkArgument(a.getShape().dim(0) == b.getShape().dim(0),
                        "batch dimensions don't match: %s @ %s", a.getShape(), b.getShape());
                batch = a.getShape().dim(0);
            } else {
                batch = aDim == 3 ? a.getShape().dim(0) : b.getShape().dim(0);
            }
            resultShape = new Shape(batch, m, n);
        }

        Storage out = Ops.matmul(a, b, resultShape);
        return Tensor.wrapResult(out, resultShape, new MatmulFn(a, b));
    }

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
