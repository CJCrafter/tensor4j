package com.cjcrafter.tensor4j.tests;

import com.cjcrafter.tensor4j.Tensor4j;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class Tensor4jTest {

    @Test
    void testNoGradRestoresState() {
        assertTrue(Tensor4j.isGradEnabled());

        try (var ignored = Tensor4j.noGrad()) {
            assertFalse(Tensor4j.isGradEnabled());

            try (var ignored2 = Tensor4j.noGrad()) {
                // nested should not re-enable
                assertFalse(Tensor4j.isGradEnabled());
            }

            // should still be disabled... e.g. state restored
            assertFalse(Tensor4j.isGradEnabled());
        }

        // now grad should finally be re-enabled
        assertTrue(Tensor4j.isGradEnabled());
    }
}
