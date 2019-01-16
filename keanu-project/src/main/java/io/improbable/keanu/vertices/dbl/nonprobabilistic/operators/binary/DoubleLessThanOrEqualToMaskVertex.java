package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;

public class DoubleLessThanOrEqualToMaskVertex extends DoubleBinaryOpVertex {

    public DoubleLessThanOrEqualToMaskVertex(DoubleVertex left, DoubleVertex right) {
        super(checkHasOneNonLengthOneShapeOrAllLengthOne(left.getShape(), right.getShape()), left, right);
    }

    @Override
    protected DoubleTensor op(DoubleTensor l, DoubleTensor r) {
        return l.getLessThanOrEqualToMask(r);
    }

    @Override
    protected PartialDerivative forwardModeAutoDifferentiation(PartialDerivative l, PartialDerivative r) {
        return null;
    }
}
