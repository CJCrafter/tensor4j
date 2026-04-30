package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Shape;
import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;

public class CatFn extends TensorFunction {

    private final int dim;
    private final int[] sizesAlongDim;

    public CatFn(int dim, int[] sizesAlongDim, Tensor... inputs) {
        super(inputs);
        this.dim = dim;
        this.sizesAlongDim = sizesAlongDim;
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        if (!gradOutput.isContiguous())
            gradOutput = gradOutput.contiguous();

        Shape outShape = gradOutput.getShape();
        int ndim = outShape.dimensions();

        int outer = 1;
        for (int d = 0; d < dim; d++) outer *= outShape.dim(d);
        int inner = 1;
        for (int d = dim + 1; d < ndim; d++) inner *= outShape.dim(d);
        int outDimStride = outShape.dim(dim) * inner;

        Tensor[] grads = new Tensor[inputs.length];
        float[] gradData = gradOutput.getData();
        int gradOff = gradOutput.getOffset();

        int cursor = 0;
        for (int k = 0; k < inputs.length; k++) {
            int sizeK = sizesAlongDim[k];
            int[] dims = inputs[k].getShape().dims();
            float[] out = new float[outer * sizeK * inner];
            int inDimStride = sizeK * inner;
            for (int o = 0; o < outer; o++) {
                int outBase = o * outDimStride + cursor * inner;
                int inBase = o * inDimStride;
                System.arraycopy(gradData, gradOff + outBase, out, inBase, sizeK * inner);
            }
            grads[k] = TensorBuilder.builder().shape(new Shape(dims)).fromArray(out);
            cursor += sizeK;
        }
        return grads;
    }
}
