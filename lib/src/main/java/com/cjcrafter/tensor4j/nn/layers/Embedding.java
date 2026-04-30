package com.cjcrafter.tensor4j.nn.layers;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;

public class Embedding extends Module {

    private final Tensor weight;
    private final int embeddingDim;

    public Embedding(int numEmbeddings, int embeddingDim) {
        this.embeddingDim = embeddingDim;
        this.weight = TensorBuilder.builder()
                .shape(numEmbeddings, embeddingDim)
                .requiresGrad()
                .randn();
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor flat = input.contiguous();
        float[] data = flat.getData();
        int off = flat.getOffset();
        int n = flat.getShape().numel();

        int[] indices = new int[n];
        for (int i = 0; i < n; i++)
            indices[i] = (int) data[off + i];

        Tensor selected = weight.indexSelect(0, indices);

        int[] inDims = input.getShape().dims();
        int[] outDims = new int[inDims.length + 1];
        System.arraycopy(inDims, 0, outDims, 0, inDims.length);
        outDims[inDims.length] = embeddingDim;
        return selected.view(outDims);
    }
}
