package com.cjcrafter.tensor4j.util;

/**
 * A version of {@link AutoCloseable} that does not throw a checked exception.
 */
public interface SafeAutoClosable extends AutoCloseable {

    /**
     * {@inheritDoc}
     */
    @Override
    void close();
}
