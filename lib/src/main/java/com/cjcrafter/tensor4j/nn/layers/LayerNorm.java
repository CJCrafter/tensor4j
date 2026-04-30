package com.cjcrafter.tensor4j.nn.layers;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;

public class LayerNorm extends Module {

    private final Tensor weight;
    private final Tensor bias;
    private final int featureSize;
    private final float eps;

    public LayerNorm(int featureSize) {
        this(featureSize, 1e-5f);
    }

    public LayerNorm(int featureSize, float eps) {
        this.featureSize = featureSize;
        this.eps = eps;
        this.weight = TensorBuilder.builder().shape(1, featureSize).requiresGrad().ones();
        this.bias = TensorBuilder.builder().shape(1, featureSize).requiresGrad().zeros();
    }

    @Override
    public Tensor forward(Tensor input) {
        int lastDim = input.getShape().dimensions() - 1;
        if (input.getShape().dim(lastDim) != featureSize)
            throw new IllegalArgumentException(
                    "LayerNorm expected last dim " + featureSize
                            + ", got shape " + input.getShape());

        float invN = 1f / featureSize;
        Tensor mean = input.sum(lastDim).mul(invN);
        Tensor centered = input.sub(mean);
        Tensor variance = centered.square().sum(lastDim).mul(invN);
        Tensor invStd = variance.add(eps).sqrt().rdiv(1f);
        Tensor normalized = centered.mul(invStd);
        return normalized.mul(weight).add(bias);
    }
}
