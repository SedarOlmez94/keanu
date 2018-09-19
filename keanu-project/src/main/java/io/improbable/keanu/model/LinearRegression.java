package io.improbable.keanu.model;

import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

public class LinearRegression implements LinearModel {

    private static final double DEFAULT_PRIOR_ON_WEIGHTS = 2.0;
    private static final VertexLabel WEIGHTS_LABEL = new VertexLabel("weights");
    private static final VertexLabel INTERCEPT_LABEL = new VertexLabel("intercept");

    private DoubleTensor x;
    private DoubleTensor y;
    private BayesianNetwork net;
    private boolean isFit;
    private double sigmaOnPrior;

    public LinearRegression(DoubleTensor x, DoubleTensor y) {
        this(x, y, DEFAULT_PRIOR_ON_WEIGHTS);
    }

    public LinearRegression(DoubleTensor x, DoubleTensor y, double sigmaOnPrior) {
        this.x = x;
        this.y = y;
        this.isFit = false;
        this.net = null;
        this.sigmaOnPrior = sigmaOnPrior;
    }

    @Override
    public LinearRegression fit() {
        DoubleVertex weights = new GaussianVertex(0.0, sigmaOnPrior).setLabel(WEIGHTS_LABEL);
        DoubleVertex intercept = new GaussianVertex(0.0, sigmaOnPrior).setLabel(INTERCEPT_LABEL);
        DoubleVertex xMu = weights.multiply(ConstantVertex.of(x));
        DoubleVertex yVertex = new GaussianVertex(xMu.plus(intercept), sigmaOnPrior);
        yVertex.observe(y);

        net = new BayesianNetwork(yVertex.getConnectedGraph());
        GradientOptimizer optimizer = GradientOptimizer.of(net);
        optimizer.maxLikelihood();
        isFit = true;
        return this;
    }

    @Override
    public DoubleTensor predict(DoubleTensor x) {
        if (isFit) {
            DoubleVertex weights = (DoubleVertex) net.getVertexByLabel(WEIGHTS_LABEL);
            DoubleVertex intercept = (DoubleVertex) net.getVertexByLabel(INTERCEPT_LABEL);
            return weights.getValue().times(x).plus(intercept.getValue());
        }
        return null;
    }

    public DoubleVertex getWeights() {
        return (DoubleVertex) net.getVertexByLabel(WEIGHTS_LABEL);
    }

    public DoubleVertex getIntercept() {
        return (DoubleVertex) net.getVertexByLabel(INTERCEPT_LABEL);
    }

    public BayesianNetwork getNet() {
        return net;
    }
}
