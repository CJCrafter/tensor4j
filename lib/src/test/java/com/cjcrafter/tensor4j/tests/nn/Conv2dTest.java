package com.cjcrafter.tensor4j.tests.nn;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.nn.layers.Conv2d;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class Conv2dTest {

    @Test
    void samePaddingPreservesSpatialDims() {
        Tensor4j.manualSeed(1);
        Conv2d conv = new Conv2d(3, 8, 3, 1, 1); // 3x3 kernel, "same" padding
        Tensor input = Tensor4j.builder().shape(2, 3, 8, 8).randn();
        Tensor output = conv.forward(input);
        assertArrayEquals(new int[]{2, 8, 8, 8}, output.getShape().dims());
    }

    @Test
    void weightAndBiasAreRegisteredParameters() {
        Tensor4j.manualSeed(2);
        Conv2d conv = new Conv2d(3, 8, 3, 1, 1);
        Map<String, Tensor> state = conv.stateDict();
        assertEquals(2, state.size());
        assertTrue(state.containsKey("weight"));
        assertTrue(state.containsKey("bias"));
        assertArrayEquals(new int[]{8, 3, 3, 3}, state.get("weight").getShape().dims());
        assertArrayEquals(new int[]{1, 8}, state.get("bias").getShape().dims());
    }

    @Test
    void noBiasOmitsBiasParameter() {
        Tensor4j.manualSeed(3);
        Conv2d conv = new Conv2d(Tensor4j.getRandom(), 3, 8, 3, 1, 1, 0.1f, false);
        Map<String, Tensor> state = conv.stateDict();
        assertEquals(1, state.size());
        assertTrue(state.containsKey("weight"));

        // forward still works with the implicit zero bias
        Tensor input = Tensor4j.builder().shape(1, 3, 8, 8).randn();
        assertDoesNotThrow(() -> conv.forward(input));
    }

    @Test
    void strideReducesSpatialDims() {
        Tensor4j.manualSeed(4);
        Conv2d conv = new Conv2d(1, 2, 3, 2, 0); // stride 2, no padding
        Tensor input = Tensor4j.builder().shape(1, 1, 6, 6).randn();
        Tensor output = conv.forward(input);
        // outH = (6 - 3)/2 + 1 = 2
        assertArrayEquals(new int[]{1, 2, 2, 2}, output.getShape().dims());
    }
}
