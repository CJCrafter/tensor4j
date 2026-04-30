package com.cjcrafter.tensor4j.io;

import com.cjcrafter.tensor4j.Tensor;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

/**
 * Reader/writer for NumPy {@code .npz} archive format.
 *
 * <p>An {@code .npz} file is a standard ZIP archive where each entry is a
 * {@code .npy} file. Entry names (minus the {@code .npy} suffix) become
 * the map keys.
 */
public final class NpzFormat {

    private NpzFormat() {}

    /**
     * Loads all arrays from a {@code .npz} file.
     *
     * @param path path to the .npz file
     * @return a map from array names to tensors, in archive order
     * @throws IOException if the file is malformed or an I/O error occurs
     */
    public static Map<String, Tensor> load(Path path) throws IOException {
        LinkedHashMap<String, Tensor> result = new LinkedHashMap<>();
        try (ZipInputStream zis = new ZipInputStream(Files.newInputStream(path))) {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                String name = entry.getName();
                if (name.endsWith(".npy")) {
                    name = name.substring(0, name.length() - 4);
                }
                result.put(name, NpyFormat.read(zis));
                zis.closeEntry();
            }
        }
        return result;
    }

    /**
     * Saves a map of tensors as a {@code .npz} file.
     *
     * @param path    path to write the .npz file
     * @param tensors map from names to tensors
     * @throws IOException if an I/O error occurs
     */
    public static void save(Path path, Map<String, Tensor> tensors) throws IOException {
        try (ZipOutputStream zos = new ZipOutputStream(Files.newOutputStream(path))) {
            for (var entry : tensors.entrySet()) {
                zos.putNextEntry(new ZipEntry(entry.getKey() + ".npy"));
                NpyFormat.write(zos, entry.getValue());
                zos.closeEntry();
            }
        }
    }
}
