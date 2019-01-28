package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import io.improbable.keanu.algorithms.variational.optimizer.*;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.util.ProgressBar;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.ToString;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

/**
 * This class can be used to construct a BOBYQA non-gradient optimizer.
 * This will use a quadratic approximation of the gradient to perform optimization without derivatives.
 *
 * @see <a href="http://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf">BOBYQA Optimizer</a>
 */
@AllArgsConstructor(access = AccessLevel.PRIVATE)
public class NonGradientOptimizer implements Optimizer {

    private final ProbabilisticGraph probabilisticGraph;

    private final NonGradientOptimizationAlgorithm nonGradientOptimizationAlgorithm;

    private final boolean checkInitialFitnessConditions;

    private final List<BiConsumer<Map<VariableReference, DoubleTensor>, Double>> onFitnessCalculations = new ArrayList<>();

    public static NonGradientOptimizerBuilder builder() {
        return new NonGradientOptimizerBuilder();
    }

    @Override
    public void addFitnessCalculationHandler(BiConsumer<Map<VariableReference, DoubleTensor>, Double> fitnessCalculationHandler) {
        this.onFitnessCalculations.add(fitnessCalculationHandler);
    }

    @Override
    public void removeFitnessCalculationHandler(BiConsumer<Map<VariableReference, DoubleTensor>, Double> fitnessCalculationHandler) {
        this.onFitnessCalculations.remove(fitnessCalculationHandler);
    }

    private void handleFitnessCalculation(Map<VariableReference, DoubleTensor> point, Double fitness) {
        for (BiConsumer<Map<VariableReference, DoubleTensor>, Double> fitnessCalculationHandler : onFitnessCalculations) {
            fitnessCalculationHandler.accept(point, fitness);
        }
    }

    private OptimizedResult optimize(FitnessFunction fitnessFunction) {

        ProgressBar progressBar = Optimizer.createFitnessProgressBar(this);

        List<? extends Variable> latentVariables = probabilisticGraph.getLatentVariables();

        if (checkInitialFitnessConditions) {
            Map<VariableReference, DoubleTensor> startingPoint = Optimizer.convertToMapPoint(latentVariables);

            double initialFitness = fitnessFunction.value(startingPoint);

            if (ProbabilityCalculator.isImpossibleLogProb(initialFitness)) {
                throw new IllegalArgumentException("Cannot start optimizer on zero probability network");
            }
        }

        OptimizedResult result = nonGradientOptimizationAlgorithm.optimize(latentVariables, fitnessFunction);

        progressBar.finish();

        return result;
    }

    @Override
    public OptimizedResult maxAPosteriori() {
        return optimize(
            new LogProbFitnessFunction(
                probabilisticGraph,
                this::handleFitnessCalculation
            )
        );
    }

    @Override
    public OptimizedResult maxLikelihood() {
        return optimize(
            new LogLikelihoodFitnessFunction(
                probabilisticGraph,
                this::handleFitnessCalculation
            )
        );
    }

    @ToString
    public static class NonGradientOptimizerBuilder {

        private ProbabilisticGraph probabilisticGraph;

        private NonGradientOptimizationAlgorithm nonGradientOptimizationAlgorithm = BOBYQA.builder().build();

        private boolean checkInitialFitnessConditions;

        NonGradientOptimizerBuilder() {
        }

        public NonGradientOptimizerBuilder bayesianNetwork(ProbabilisticGraph probabilisticGraph) {
            this.probabilisticGraph = probabilisticGraph;
            return this;
        }

        public NonGradientOptimizerBuilder algorithm(NonGradientOptimizationAlgorithm nonGradientOptimizationAlgorithm) {
            this.nonGradientOptimizationAlgorithm = nonGradientOptimizationAlgorithm;
            return this;
        }

        public NonGradientOptimizerBuilder checkInitialFitnessConditions(boolean check) {
            this.checkInitialFitnessConditions = check;
            return this;
        }

        public NonGradientOptimizer build() {
            if (probabilisticGraph == null) {
                throw new IllegalStateException("Cannot build optimizer without specifying network to optimize.");
            }
            return new NonGradientOptimizer(
                probabilisticGraph,
                nonGradientOptimizationAlgorithm,
                checkInitialFitnessConditions
            );
        }
    }
}