package com.cjcrafter.tensor4j.nn.layers;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.io.NpzFormat;

import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.file.Path;
import java.util.*;

/**
 * Base class for all neural network modules.
 *
 * <p>Parameters ({@link Tensor} fields) and child modules ({@link Module}
 * fields) are discovered automatically via reflection. Arrays,
 * {@link Collection Collections}, and {@link Map Maps} of these types are
 * expanded using indices or keys as names.
 */
public abstract class Module {

    /**
     * Does the full forward computation.
     *
     * @param input The input <code>x</code> to the neural network.
     * @return The output <code>y_hat</code> of the neural network.
     */
    public abstract Tensor forward(Tensor input);

    public final void eval() {
        train(false);
    }

    public final void train() {
        train(true);
    }

    public void train(boolean mode) {
        // by default, this method should not be implemented by a subclass.
        // Some modules, like Dropout, need to behave differently depending
        // on whether you are training or evaluating.
    }

    /**
     * Returns all parameters in this module and its children.
     *
     * @return All parameters of the neural network.
     */
    public Collection<Tensor> parameters() {
        return stateDict().values();
    }


    /**
     * Recursively collects all named parameters using dot-separated paths
     * (e.g. {@code "0.weight"}), discovered via reflection.
     *
     * @return An ordered map from dotted parameter name to tensor.
     */
    public final Map<String, Tensor> stateDict() {
        LinkedHashMap<String, Tensor> state = new LinkedHashMap<>();
        collectStateDict("", getClass(), state);
        return Collections.unmodifiableMap(state);
    }

    private void collectStateDict(String prefix, Class<?> clazz, Map<String, Tensor> state) {
        if (clazz == null || clazz == Module.class) return;
        collectStateDict(prefix, clazz.getSuperclass(), state);

        for (Field field : clazz.getDeclaredFields()) {
            field.setAccessible(true);
            Object value;
            try {
                value = field.get(this);
            } catch (IllegalAccessException e) {
                continue;
            }
            switch (value) {
                case Tensor t -> state.put(prefix + field.getName(), t);
                case Module m -> m.collectStateDict(prefix + field.getName() + ".", m.getClass(), state);
                case Object[] arr -> collectFromArray(prefix, arr, state);
                case Collection<?> col -> collectFromArray(prefix, col.toArray(), state);
                case Map<?, ?> map -> {
                    for (var entry : map.entrySet()) {
                        String key = prefix + entry.getKey().toString();
                        Object v = entry.getValue();
                        if (v instanceof Tensor t) {
                            state.put(key, t);
                        } else if (v instanceof Module m) {
                            m.collectStateDict(key + ".", m.getClass(), state);
                        }
                    }
                }
                case null, default -> {
                }
            }

        }
    }

    private void collectFromArray(String prefix, Object[] arr, Map<String, Tensor> state) {
        for (int i = 0; i < arr.length; i++) {
            Object item = arr[i];
            if (item instanceof Tensor t) {
                state.put(prefix + i, t);
            } else if (item instanceof Module m) {
                m.collectStateDict(prefix + i + ".", m.getClass(), state);
            }
        }
    }

    /**
     * Loads parameter values from a state dict by copying data into the
     * existing backing arrays of this module's parameters.
     *
     * @param stateDict A map from dotted parameter names to source tensors.
     * @throws IllegalArgumentException if a required key is missing or shapes don't match.
     */
    public final void loadStateDict(Map<String, Tensor> stateDict) {
        Map<String, Tensor> current = stateDict();
        for (var entry : current.entrySet()) {
            Tensor source = stateDict.get(entry.getKey());
            if (source == null)
                throw new IllegalArgumentException("Missing key in state dict: " + entry.getKey());

            Tensor target = entry.getValue();
            int targetNumel = target.getShape().numel();
            int sourceNumel = source.getShape().numel();
            if (targetNumel != sourceNumel)
                throw new IllegalArgumentException(
                    "Shape mismatch for '" + entry.getKey() + "': expected " + targetNumel
                    + " elements, got " + sourceNumel);

            Tensor src = source.contiguous();
            System.arraycopy(src.getData(), src.getOffset(),
                             target.getData(), target.getOffset(), targetNumel);
        }
    }

    public void load(Path path) throws IOException {
        loadStateDict(NpzFormat.load(path));
    }

    public void save(Path path) throws IOException {
        NpzFormat.save(path, stateDict());
    }
}
