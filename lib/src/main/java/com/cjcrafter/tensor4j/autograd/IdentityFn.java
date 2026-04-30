package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Tensor;

public class IdentityFn extends TensorFunction {

    public IdentityFn(Tensor input) {
        super(input);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        return new Tensor[] {
                gradOutput.clone()
        };
    }
}
