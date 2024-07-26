package org.apache.seatunnel.connectors.seatunnel.milvus.utils;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;

import org.tensorflow.ndarray.buffer.ByteDataBuffer;
import org.tensorflow.types.TBfloat16;

public class DataConverter {

    public List<Float> convertFloatVector(Object value) {
        List<Float> float_vector = new ArrayList<>();
        for (Object o : (Object[]) value) {
            float_vector.add(Float.parseFloat(o.toString()));
        }
        return float_vector;
    }

    public ByteBuffer convertFloat16Vector(Object value) {
        List<Float> vector = convertFloatVector(value);
        ByteBuffer buf = ByteBuffer.allocate(2 * vector.size());
        buf.order(ByteOrder.LITTLE_ENDIAN); // milvus server stores fp16/bf16 vector as little endian
        for (Float val : vector) {
            short bf16 = floatToBf16(val);
            buf.putShort(bf16);
        }
        return buf;
    }

    public List<Integer> convertBinaryVector(Object value) {
        List<Integer> binary_vector = new ArrayList<>();
        for (Object o : (Object[]) value) {
            binary_vector.add(Integer.parseInt(o.toString()));
        }
        return binary_vector;
    }

    public ByteBuffer convertBFloat16Vector(Object value) {
        Object[] objects = (Object[]) value;
        float[] bfloat16_array = new float[objects.length];

        for (int i = 0; i < objects.length; i++) {
            bfloat16_array[i] = Float.parseFloat(objects[i].toString());
        }

        TBfloat16 vectors = TBfloat16.vectorOf(bfloat16_array);
        return encodeTensorBF16Vector(vectors);
    }

    public static short floatToBf16(float input) {
        int bits = Float.floatToIntBits(input);
        int lsb = (bits >> 16) & 1;
        int roundingBias = 0x7fff + lsb;
        bits += roundingBias;
        return (short) (bits >> 16);
    }
    public static ByteBuffer encodeTensorBF16Vector(TBfloat16 vector) {
        ByteDataBuffer tensorBuf = vector.asRawTensor().data();
        ByteBuffer buf = ByteBuffer.allocate((int)tensorBuf.size());
        for (long i = 0; i < tensorBuf.size(); i++) {
            buf.put(tensorBuf.getByte(i));
        }
        return buf;
    }

    public SortedMap<Long, Float> convertSparseVector(Object value) {
        // Assuming value is already a Map
        Map<?, ?> inputMap = (Map<?, ?>) value;

        // Create a TreeMap to store the sorted map
        SortedMap<Long, Float> sparseVector = new TreeMap<>();

        // Iterate over the entries of the map
        for (Map.Entry<?, ?> entry : inputMap.entrySet()) {
            Long key = Long.parseLong(entry.getKey().toString());
            Float val = Float.parseFloat(entry.getValue().toString());

            // Put the key-value pair into the TreeMap
            sparseVector.put(key, val);
        }

        return sparseVector;
    }
}
