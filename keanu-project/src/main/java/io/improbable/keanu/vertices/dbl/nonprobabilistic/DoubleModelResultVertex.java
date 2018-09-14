package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.ModelResultVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple.ModelVertex;
import io.improbable.keanu.vertices.Vertex;

public class DoubleModelResultVertex extends DoubleVertex implements NonProbabilistic<DoubleTensor> {

    private ModelVertex model;
    private VertexLabel label;
    private boolean hasValue;

    public DoubleModelResultVertex(ModelVertex model, VertexLabel label) {
        this.model = model;
        this.label = label;
        this.hasValue = false;
        setParents((Vertex) model);
    }

    @Override
    public DoubleTensor getValue() {
        if (!hasValue) {
            model.calculate();
        }
        return model.getDoubleModelOutputValue(label);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return model.getDoubleModelOutputValue(label);
    }

    @Override
    public DoubleTensor calculate() {
        hasValue = true;
        return model.getDoubleModelOutputValue(label);
    }

}
