package com.cjcrafter.tensor4j.nn.losses;

import com.cjcrafter.tensor4j.Tensor;

public record CrossEntropyLoss(Reduction reduction, int dim) implements Loss {

    public CrossEntropyLoss() {
        this(Reduction.MEAN, -1);
    }

    public CrossEntropyLoss(Reduction reduction) {
        this(reduction, -1);
    }

    @Override
    public Tensor forward(Tensor input, Tensor target) {
        int resolvedDim = (dim >= 0) ? dim : input.getShape().dimensions() + dim;
        Tensor logProbs = input.logSoftmax(resolvedDim);
        Tensor perEntry = target.mul(logProbs).sum(resolvedDim).mul(-1f);
        return reduction.reduce(perEntry);
    }
}
