package com.cjcrafter.tensor4j.nn.optimizers;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;

import java.util.Collection;

public class SGD implements Optimizer {

    private final Collection<Tensor> params;
    private final float lr;

    public SGD(Collection<Tensor> params, float lr) {
        this.params = params;
        this.lr = lr;
    }

    @Override
    public void step() {
        if (Tensor4j.isGradEnabled())
            throw new IllegalStateException("Cannot step while grad is enabled... see Tensor4j.noGrad()");

        for (Tensor param : params) {
            Tensor grad = param.getGrad();
            if (grad == null)
                continue;

            grad.mul_(lr);
            param.sub_(grad);
        }
    }

    @Override
    public void zeroGrad() {
        for (Tensor p : params) {
            p.zeroGrad();
        }
    }
}
