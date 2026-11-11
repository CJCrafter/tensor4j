package com.cjcrafter.tensor4j.nn.layers;

import com.cjcrafter.tensor4j.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Chains a sequence of modules, feeding each one's output into the next.
 */
public class Sequential extends Module {

    private final List<Module> modules;

    /**
     * Creates a module that applies {@code modules} in order.
     * @param modules the modules to apply, in order
     */
    public Sequential(Module... modules) {
        this.modules = new ArrayList<>(List.of(modules));
    }

    /**
     * Appends {@code module} to the end of this sequence.
     *
     * @return This sequence, for chaining.
     * @param module the module to append
     */
    public Sequential add(Module module) {
        modules.add(module);
        return this;
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
