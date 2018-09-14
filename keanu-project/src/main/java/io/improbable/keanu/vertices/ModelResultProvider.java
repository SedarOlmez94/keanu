package io.improbable.keanu.vertices;

import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple.ModelVertex;

public interface ModelResultProvider<T> {

    ModelVertex<T> getModel();

    VertexLabel getLabel();

    default T getValue() {
        if (!getModel().hasCalculated()) {
            getModel().calculate();
        }
        return getModel().getModelOutputValue(getLabel());
    }
}
