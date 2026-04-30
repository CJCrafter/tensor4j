package com.cjcrafter.tensor4j.tests.autograd;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DetachTest {

    @Test
    void testDetachedHasNoGrad() {
        Tensor4j.manualSeed(2187);
        Tensor a = Tensor4j.builder().shape(3, 3).requiresGrad().randn();
        Tensor d = a.detach();

        assertTrue(a.requiresGrad());
        assertFalse(d.requiresGrad());
        assertNull(d.getGradFn());
    }

    @Test
    void testDetachSharesData() {
        Tensor4j.manualSeed(2187);
        Tensor a = Tensor4j.builder().shape(2, 3).requiresGrad().randn();
        Tensor d = a.detach();

        assertSame(a.getData(), d.getData());
        assertEquals(a.getShape(), d.getShape());
        assertEquals(a.getOffset(), d.getOffset());
    }

    @Test
    void testDetachStopsBackprop() {
        Tensor4j.manualSeed(2187);
        Tensor a = Tensor4j.builder().shape(3, 3).requiresGrad().randn();
        Tensor b = Tensor4j.builder().shape(3, 3).requiresGrad().randn();

        Tensor loss = a.detach().mul(b).sum();
        try (var ignored = Tensor4j.noGrad()) {
            loss.backward();
        }

        assertNull(a.getGrad(), "grad should not flow through detached path");
        assertNotNull(b.getGrad(), "grad should still flow to non-detached operand");
    }

    @Test
    void testDetachMixedPath() {
        Tensor4j.manualSeed(2187);
        Tensor a = Tensor4j.builder().shape(2, 2).requiresGrad().randn();

        Tensor viaGrad = a.mul(2f);
        Tensor viaDetach = a.detach().mul(3f);
        Tensor loss = viaGrad.add(viaDetach).sum();
        try (var ignored = Tensor4j.noGrad()) {
            loss.backward();
        }

        for (float g : a.getGrad().getData()) {
            assertEquals(2f, g, 1e-6f,
                    "only the non-detached mul(2f) branch should contribute");
        }
    }
}
