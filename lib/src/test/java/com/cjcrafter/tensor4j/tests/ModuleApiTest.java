package com.cjcrafter.tensor4j.tests;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.nn.layers.Linear;
import com.cjcrafter.tensor4j.nn.layers.Module;
import com.cjcrafter.tensor4j.nn.layers.ReLU;
import com.cjcrafter.tensor4j.nn.layers.Sequential;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ModuleApiTest {

    @Test
    void sequentialAddAppendsAndChains() {
        Tensor4j.manualSeed(1);
        Sequential net = new Sequential(new Linear(4, 8));
        Sequential same = net.add(new ReLU()).add(new Linear(8, 2));
        assertSame(net, same);
        Tensor out = net.forward(Tensor4j.builder().shape(3, 4).randn());
        assertArrayEquals(new int[]{3, 2}, out.getShape().dims());
        assertTrue(net.stateDict().containsKey("2.weight"));
    }

    @Test
    void zeroGradClearsAllParameterGradients() {
        Tensor4j.manualSeed(2);
        Linear lin = new Linear(3, 2);
        lin.forward(Tensor4j.builder().shape(4, 3).randn()).sum().backward();
        for (Tensor p : lin.parameters()) assertTrue(p.hasGrad());

        lin.zeroGrad();
        for (Tensor p : lin.parameters()) assertFalse(p.hasGrad());
    }

    @Test
    void forwardAllDefaultsToSingleOutput() {
        Tensor4j.manualSeed(3);
        Linear lin = new Linear(3, 2);
        Tensor input = Tensor4j.builder().shape(4, 3).randn();
        Tensor[] outs = lin.forwardAll(input);
        assertEquals(1, outs.length);
        assertArrayEquals(lin.forward(input).toFloatArray(), outs[0].toFloatArray(), 1e-6f);
    }

    @Test
    void multiHeadModuleReturnsAllOutputs() {
        Tensor4j.manualSeed(4);
        Module twoHeads = new Module() {
            private final Linear policy = new Linear(4, 3);
            private final Linear value = new Linear(4, 1);

            @Override
            public Tensor forward(Tensor input) {
                return forwardAll(input)[0];
            }

            @Override
            public Tensor[] forwardAll(Tensor input) {
                return new Tensor[]{policy.forward(input), value.forward(input).tanh()};
            }
        };
        Tensor[] outs = twoHeads.forwardAll(Tensor4j.builder().shape(2, 4).randn());
        assertArrayEquals(new int[]{2, 3}, outs[0].getShape().dims());
        assertArrayEquals(new int[]{2, 1}, outs[1].getShape().dims());
        assertTrue(twoHeads.stateDict().containsKey("policy.weight"));
        assertTrue(twoHeads.stateDict().containsKey("value.bias"));
    }
}
