package com.cjcrafter.tensor4j.tests.io;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.io.NpzFormat;
import com.cjcrafter.tensor4j.nn.layers.Linear;
import com.cjcrafter.tensor4j.nn.layers.ReLU;
import com.cjcrafter.tensor4j.nn.layers.Sequential;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class TensorSaveTest {

    private static final float EPS = 1e-6f;

    @Test
    void stateDictKeys() {
        Tensor4j.manualSeed(42);
        Sequential model = new Sequential(
                new Linear(784, 128),
                new ReLU(),
                new Linear(128, 10)
        );

        Map<String, Tensor> state = model.stateDict();

        assertTrue(state.containsKey("0.weight"));
        assertTrue(state.containsKey("0.bias"));
        assertTrue(state.containsKey("2.weight"));
        assertTrue(state.containsKey("2.bias"));
        assertEquals(4, state.size());

        // Verify shapes
        assertArrayEquals(new int[]{784, 128}, state.get("0.weight").getShape().dims());
        assertArrayEquals(new int[]{1, 128}, state.get("0.bias").getShape().dims());
        assertArrayEquals(new int[]{128, 10}, state.get("2.weight").getShape().dims());
        assertArrayEquals(new int[]{1, 10}, state.get("2.bias").getShape().dims());
    }

    @Test
    void saveAndLoadRoundtrip(@TempDir Path tempDir) throws IOException {
        Tensor4j.manualSeed(42);
        Sequential model = new Sequential(
                new Linear(16, 8),
                new ReLU(),
                new Linear(8, 4)
        );

        Path file = tempDir.resolve("model.npz");
        model.save(file);

        // Create a fresh model with same architecture and different weights
        Tensor4j.manualSeed(99);
        Sequential fresh = new Sequential(
                new Linear(16, 8),
                new ReLU(),
                new Linear(8, 4)
        );

        // Verify weights are different before load
        Map<String, Tensor> originalState = model.stateDict();
        Map<String, Tensor> freshState = fresh.stateDict();
        boolean allSame = true;
        for (String key : originalState.keySet()) {
            Tensor a = originalState.get(key);
            Tensor b = freshState.get(key);
            if (Math.abs(a.get(0, 0) - b.get(0, 0)) > EPS) {
                allSame = false;
                break;
            }
        }
        assertFalse(allSame, "Fresh model should have different weights");

        // Load saved weights
        fresh.load(file);

        // Verify weights match
        Map<String, Tensor> loadedState = fresh.stateDict();
        for (String key : originalState.keySet()) {
            Tensor expected = originalState.get(key);
            Tensor actual = loadedState.get(key);
            int numel = expected.getShape().numel();
            for (int i = 0; i < numel; i++) {
                assertEquals(expected.getData()[expected.getOffset() + i],
                             actual.getData()[actual.getOffset() + i], EPS,
                             "Mismatch in " + key + " at index " + i);
            }
        }
    }

    @Test
    void npzTensorRoundtrip(@TempDir Path tempDir) throws IOException {
        Tensor4j.manualSeed(42);
        Tensor a = Tensor4j.builder().shape(3, 4).randn();
        Tensor b = Tensor4j.builder().shape(2, 5).randn();

        Path file = tempDir.resolve("tensors.npz");
        NpzFormat.save(file, Map.of("a", a, "b", b));

        Map<String, Tensor> loaded = NpzFormat.load(file);
        assertTrue(loaded.containsKey("a"));
        assertTrue(loaded.containsKey("b"));

        assertArrayEquals(new int[]{3, 4}, loaded.get("a").getShape().dims());
        assertArrayEquals(new int[]{2, 5}, loaded.get("b").getShape().dims());

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                assertEquals(a.get(i, j), loaded.get("a").get(i, j), EPS);
    }
}
