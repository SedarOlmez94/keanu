package io.improbable.keanu.algorithms.prior;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.ProbabilisticModel;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.graphtraversal.TopologicalSort;
import io.improbable.keanu.vertices.Vertex;

public class Prior {

    private Prior() {
    }

    /**
     * Samples from a Bayesian Network that only contains prior information. No observations can have been made.
     * Samples are taken by calculating a linear ordering of the network and cascading the sampled values
     * through the network in priority order.
     *
     * @param model the prior bayesian network to sample from
     * @param fromVertices the vertices to sample from
     * @param sampleCount the number of samples to take
     * @return prior samples of a bayesian network
     */
    public static NetworkSamples sample(ProbabilisticModel model,
                                        List<? extends Vertex> fromVertices,
                                        int sampleCount) {
        return sample(model, fromVertices, sampleCount, KeanuRandom.getDefaultRandom());
    }

    public static NetworkSamples sample(ProbabilisticModel model,
                                        List<? extends Vertex> fromVertices,
                                        int sampleCount,
                                        KeanuRandom random) {

        List<? extends Variable> sorted = model.sort(model.getLatentVariables());
        Map<Long, List> samplesByVertex = new HashMap<>();

        for (int sampleNum = 0; sampleNum < sampleCount; sampleNum++) {
            nextSample(sorted, random);
            takeSamples(samplesByVertex, fromVertices);
        }

        return new NetworkSamples(samplesByVertex, sampleCount);
    }

    private static void nextSample(List<? extends Variable> topologicallySorted, KeanuRandom random) {
        for (Variable variable: topologicallySorted) {
            setAndCascadeFromSample(variable, random);
        }
    }

    private static void setAndCascadeFromSample(Variable variable, KeanuRandom random) {
        variable.setAndCascade(variable.sample(random));
    }

    private static void takeSamples(Map<Long, List> samples, List<? extends Vertex> fromVertices) {
        fromVertices.forEach(vertex -> addSampleForVertex(vertex, samples));
    }

    private static void addSampleForVertex(Vertex vertex, Map<Long, List> samples) {
        List samplesForVertex = samples.computeIfAbsent(vertex.getId(), v -> new ArrayList<>());
        samplesForVertex.add(vertex.getValue());
    }
}