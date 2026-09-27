package com.cjcrafter.tensor4j.tests.nn;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.nn.layers.Linear;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ModuleErrorPathsTest {

    @Test
    void loadStateDictThrowsOnMissingKey() {
        Tensor4j.manualSeed(1);
        Linear model = new Linear(4, 2);
        Map<String, Tensor> partial = new HashMap<>(model.stateDict());
        partial.remove("bias");
        assertThrows(IllegalArgumentException.class, () -> model.loadStateDict(partial));
    }

    @Test
    void loadStateDictThrowsOnShapeMismatch() {
        Tensor4j.manualSeed(1);
        Linear model = new Linear(4, 2);
        Map<String, Tensor> mismatched = new HashMap<>(model.stateDict());
        mismatched.put("weight", Tensor4j.builder().shape(1, 1).zeros());
        assertThrows(IllegalArgumentException.class, () -> model.loadStateDict(mismatched));
    }

    @Test
    void parametersMatchesStateDictValues() {
        Tensor4j.manualSeed(1);
        Linear model = new Linear(4, 2);
        assertEquals(model.stateDict().values().size(), model.parameters().size());
        assertTrue(model.parameters().containsAll(model.stateDict().values()));
    }
}
