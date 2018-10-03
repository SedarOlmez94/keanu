package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class MatrixMultiplicationVertex extends DoubleBinaryOpVertex {
    /**
     * Matrix multiplies one vertex by another. C = AB
     *
     * @param left  vertex A
     * @param right vertex B
     */

    public MatrixMultiplicationVertex(DoubleVertex left, DoubleVertex right) {
        super(getResultingShape(left.getShape(), right.getShape()),
            left, right);
    }

    @Override
    protected DoubleTensor op(DoubleTensor l, DoubleTensor r) {
        return l.matrixMultiply(r);
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {

        PartialDerivatives partialsLeft = PartialDerivatives
            .matrixMultiplyAlongWrtDimensions(
                derivativeOfOutputsWithRespectToSelf,
                right.getValue(),
                true
            );

        PartialDerivatives partialsRight = PartialDerivatives
            .matrixMultiplyAlongWrtDimensions(
                derivativeOfOutputsWithRespectToSelf,
                left.getValue(),
                false
            );

        Map<Vertex, PartialDerivatives> partials = new HashMap<>();
        partials.put(left, partialsLeft);
        partials.put(right, partialsRight);

        return partials;
    }

    @Override
    protected PartialDerivatives forwardModeAutoDifferentiation(PartialDerivatives l, PartialDerivatives r) {

        // dc = A * db + da * B;
        PartialDerivatives thisInfMultiplied;
        PartialDerivatives thatInfMultiplied;

        if (l.isEmpty()) {
            thisInfMultiplied = PartialDerivatives.OF_CONSTANT;
        } else {
            thisInfMultiplied = PartialDerivatives.matrixMultiplyAlongOfDimensions(l, right.getValue(), true);
        }

        if (r.isEmpty()) {
            thatInfMultiplied = PartialDerivatives.OF_CONSTANT;
        } else {
            thatInfMultiplied = PartialDerivatives.matrixMultiplyAlongOfDimensions(r, left.getValue(), false);
        }

        return thisInfMultiplied.add(thatInfMultiplied);
    }

    private static int[] getResultingShape(int[] left, int[] right) {
        if (left.length != 2 || right.length != 2) {
            throw new IllegalArgumentException("Matrix multiply must be used on matrices");
        }

        if (left[1] != right[0]) {
            throw new IllegalArgumentException("Can not multiply matrices of shapes " + Arrays.toString(left) + " X " + Arrays.toString(right));
        }

        return new int[]{left[0], right[1]};
    }
}
