package com.cjcrafter.tensor4j.nn.layers;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;

/**
 * A lookup table that maps integer indices to learnable dense vectors, e.g. for word
 * embeddings. Input elements are truncated to {@code int} indices into the weight table.
 */
public class Embedding extends Module {

    private final Tensor weight;
    private final int embeddingDim;

    /**
     * Creates an embedding table with {@code numEmbeddings} rows of {@code embeddingDim}
     * randomly initialized values each.
     */
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
        int n = flat.getShape().numel();
        float[] data = flat.toFloatArray();

        int[] indices = new int[n];
        for (int i = 0; i < n; i++)
            indices[i] = (int) data[i];

        Tensor selected = weight.indexSelect(0, indices);

        int[] inDims = input.getShape().dims();
        int[] outDims = new int[inDims.length + 1];
        System.arraycopy(inDims, 0, outDims, 0, inDims.length);
        outDims[inDims.length] = embeddingDim;
        return selected.view(outDims);
    }
}
