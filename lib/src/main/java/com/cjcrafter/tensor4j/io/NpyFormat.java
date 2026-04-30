package com.cjcrafter.tensor4j.io;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Low-level reader/writer for the NumPy {@code .npy} binary format.
 *
 * <p>Supports float32 arrays only. Handles both little-endian ({@code <f4})
 * and big-endian ({@code >f4}) data, and automatically reshapes 1D arrays
 * to 2D since tensor4j requires minimum 2D tensors.
 *
 * @see <a href="https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html">NumPy format spec</a>
 */
public final class NpyFormat {

    private static final byte[] MAGIC = {(byte) 0x93, 'N', 'U', 'M', 'P', 'Y'};
    private static final Pattern DESCR_PATTERN = Pattern.compile("'descr'\\s*:\\s*'([^']*)'");
    private static final Pattern FORTRAN_PATTERN = Pattern.compile("'fortran_order'\\s*:\\s*(True|False)");
    private static final Pattern SHAPE_PATTERN = Pattern.compile("'shape'\\s*:\\s*\\(([^)]*)\\)");

    private NpyFormat() {}

    /**
     * Reads a {@code .npy} formatted stream into a Tensor.
     *
     * @param in the input stream containing .npy data
     * @return a Tensor with the array's data and shape
     * @throws IOException if the stream is malformed or uses an unsupported dtype
     */
    public static Tensor read(InputStream in) throws IOException {
        DataInputStream din = new DataInputStream(in);

        // Verify magic
        byte[] magic = new byte[6];
        din.readFully(magic);
        for (int i = 0; i < 6; i++) {
            if (magic[i] != MAGIC[i])
                throw new IOException("Not a .npy file");
        }

        // Version
        int major = din.readUnsignedByte();
        int minor = din.readUnsignedByte();

        // Header length (little-endian)
        int headerLen;
        if (major == 1) {
            // 2-byte little-endian
            int lo = din.readUnsignedByte();
            int hi = din.readUnsignedByte();
            headerLen = lo | (hi << 8);
        } else if (major == 2) {
            // 4-byte little-endian
            byte[] hb = new byte[4];
            din.readFully(hb);
            headerLen = ByteBuffer.wrap(hb).order(ByteOrder.LITTLE_ENDIAN).getInt();
        } else {
            throw new IOException("Unsupported .npy version: " + major + "." + minor);
        }

        // Read header string
        byte[] headerBytes = new byte[headerLen];
        din.readFully(headerBytes);
        String header = new String(headerBytes, java.nio.charset.StandardCharsets.US_ASCII).trim();

        // Parse descr
        Matcher descrMatch = DESCR_PATTERN.matcher(header);
        if (!descrMatch.find())
            throw new IOException("Missing 'descr' in .npy header");
        String descr = descrMatch.group(1);

        // Parse byte order and dtype kind/size from descr
        char orderChar = descr.charAt(0);
        String typeStr = descr.substring(1); // e.g. "f4", "u1", "i4", "f8"

        ByteOrder byteOrder = switch (orderChar) {
            case '<', '=' -> ByteOrder.LITTLE_ENDIAN;
            case '>' -> ByteOrder.BIG_ENDIAN;
            case '|' -> ByteOrder.LITTLE_ENDIAN; // single-byte types, order irrelevant
            default -> throw new IOException("Unknown byte order in dtype: " + descr);
        };

        // Parse fortran_order
        Matcher fortranMatch = FORTRAN_PATTERN.matcher(header);
        boolean fortranOrder = fortranMatch.find() && fortranMatch.group(1).equals("True");

        // Parse shape
        Matcher shapeMatch = SHAPE_PATTERN.matcher(header);
        if (!shapeMatch.find())
            throw new IOException("Missing 'shape' in .npy header");
        String shapeStr = shapeMatch.group(1).trim();

        int[] dims;
        if (shapeStr.isEmpty()) {
            dims = new int[]{1, 1};
        } else {
            String[] parts = shapeStr.split("\\s*,\\s*");
            // Filter out empty strings from trailing commas like "(5,)"
            int count = 0;
            for (String p : parts) {
                if (!p.isEmpty())
                    count++;
            }
            dims = new int[count];
            int idx = 0;
            for (String p : parts) {
                if (!p.isEmpty()) dims[idx++] = Integer.parseInt(p);
            }
        }

        // Enforce minimum 2D... convert vectors to column vectors
        if (dims.length == 1) {
            dims = new int[]{dims[0], 1};
        }

        int numel = 1;
        for (int d : dims)
            numel *= d;

        // Read raw bytes and cast to float32
        float[] data = switch (typeStr) {
            case "f4" -> {
                byte[] raw = new byte[numel * 4];
                din.readFully(raw);
                float[] f = new float[numel];
                ByteBuffer.wrap(raw).order(byteOrder).asFloatBuffer().get(f);
                yield f;
            }
            case "f8" -> {
                byte[] raw = new byte[numel * 8];
                din.readFully(raw);
                float[] f = new float[numel];
                var dbuf = ByteBuffer.wrap(raw).order(byteOrder).asDoubleBuffer();
                for (int i = 0; i < numel; i++) f[i] = (float) dbuf.get(i);
                yield f;
            }
            case "i1", "b1" -> {
                byte[] raw = new byte[numel];
                din.readFully(raw);
                float[] f = new float[numel];
                for (int i = 0; i < numel; i++) f[i] = raw[i];
                yield f;
            }
            case "u1" -> {
                byte[] raw = new byte[numel];
                din.readFully(raw);
                float[] f = new float[numel];
                for (int i = 0; i < numel; i++) f[i] = raw[i] & 0xFF;
                yield f;
            }
            case "i2" -> {
                byte[] raw = new byte[numel * 2];
                din.readFully(raw);
                float[] f = new float[numel];
                var sbuf = ByteBuffer.wrap(raw).order(byteOrder).asShortBuffer();
                for (int i = 0; i < numel; i++) f[i] = sbuf.get(i);
                yield f;
            }
            case "i4" -> {
                byte[] raw = new byte[numel * 4];
                din.readFully(raw);
                float[] f = new float[numel];
                var ibuf = ByteBuffer.wrap(raw).order(byteOrder).asIntBuffer();
                for (int i = 0; i < numel; i++) f[i] = ibuf.get(i);
                yield f;
            }
            case "i8" -> {
                byte[] raw = new byte[numel * 8];
                din.readFully(raw);
                float[] f = new float[numel];
                var lbuf = ByteBuffer.wrap(raw).order(byteOrder).asLongBuffer();
                for (int i = 0; i < numel; i++) f[i] = lbuf.get(i);
                yield f;
            }
            default -> throw new IOException("Unsupported dtype: " + descr);
        };

        // Handle fortran order by reversing dims and making contiguous
        if (fortranOrder) {
            int[] reversed = new int[dims.length];
            for (int i = 0; i < dims.length; i++)
                reversed[i] = dims[dims.length - 1 - i];
            return TensorBuilder.builder()
                    .shape(reversed)
                    .fromArray(data)
                    .transpose()
                    .contiguous();
        }

        return TensorBuilder.builder()
                .shape(dims)
                .fromArray(data);
    }

    /**
     * Writes a Tensor to a stream in {@code .npy} format (version 1.0, float32, little-endian, C-order).
     *
     * @param out    the output stream
     * @param tensor the tensor to write
     * @throws IOException if an I/O error occurs
     */
    public static void write(OutputStream out, Tensor tensor) throws IOException {
        Tensor t = tensor.contiguous();
        int[] dims = t.getShape().dims();

        // Build header dict
        StringBuilder shapeStr = new StringBuilder("(");
        for (int i = 0; i < dims.length; i++) {
            shapeStr.append(dims[i]);
            if (i < dims.length - 1) shapeStr.append(", ");
            else if (dims.length == 1) shapeStr.append(",");
        }
        shapeStr.append(")");

        String headerDict = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shapeStr + ", }";

        // Pad header to 64-byte alignment: magic(6) + version(2) + header_len(2) + header = multiple of 64
        int preambleLen = 6 + 2 + 2; // magic + version + header_len
        int totalUnpadded = preambleLen + headerDict.length() + 1; // +1 for trailing newline
        int padding = (64 - (totalUnpadded % 64)) % 64;
        String paddedHeader = headerDict + " ".repeat(padding) + "\n";

        DataOutputStream dout = new DataOutputStream(out);

        // Magic
        dout.write(MAGIC);

        // Version 1.0
        dout.writeByte(1);
        dout.writeByte(0);

        // Header length (little-endian 2 bytes)
        int headerLen = paddedHeader.length();
        dout.writeByte(headerLen & 0xFF);
        dout.writeByte((headerLen >> 8) & 0xFF);

        // Header
        dout.writeBytes(paddedHeader);

        // Data (little-endian float32)
        float[] data = t.getData();
        int offset = t.getOffset();
        int numel = t.getShape().numel();
        ByteBuffer buf = ByteBuffer.allocate(numel * 4).order(ByteOrder.LITTLE_ENDIAN);
        buf.asFloatBuffer().put(data, offset, numel);
        dout.write(buf.array());

        dout.flush();
    }
}
