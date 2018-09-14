package io.improbable.keanu.model;

import io.improbable.keanu.vertices.ModelResultProvider;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple.ModelVertex;

public class ModelResult<T> implements ModelResultProvider<T> {

    private final ModelVertex<T> model;
    private final VertexLabel label;

    public ModelResult(ModelVertex<T> model, VertexLabel label) {
        this.model = model;
        this.label = label;
    }

    public T sample(KeanuRandom random) {
        return getModel().getModelOutputValue(getLabel());
    }

    public T calculate() {
        return sample(KeanuRandom.getDefaultRandom());
    }

    @Override
    public ModelVertex<T> getModel() {
        return model;
    }

    @Override
    public VertexLabel getLabel() {
        return label;
    }
}
