package com.cjcrafter.tensor4j.nn.layers;

import com.cjcrafter.tensor4j.Tensor;

public class Sequential extends Module {

    private final Module[] modules;

    public Sequential(Module... modules) {
        this.modules = modules;
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor x = input;
        for (Module m : modules) {
            x = m.forward(x);
        }
        return x;
    }

    @Override
    public void train(boolean mode) {
        for (Module module : modules) {
            module.train(mode);
        }
    }
}
