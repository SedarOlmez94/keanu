package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import io.improbable.keanu.algorithms.ProbabilisticModel;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.variational.optimizer.Optimizer;
import io.improbable.keanu.util.status.StatusBar;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.SimpleBounds;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.stream.Collectors;

import static io.improbable.keanu.algorithms.variational.optimizer.Optimizer.getAsDoubleTensors;
import static org.apache.commons.math3.optim.nonlinear.scalar.GoalType.MAXIMIZE;

/**
 * This class can be used to construct a BOBYQA non-gradient optimizer.
 * This will use a quadratic approximation of the gradient to perform optimization without derivatives.
 *
 * @see <a href="http://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf">BOBYQA Optimizer</a>
 */
@AllArgsConstructor(access = AccessLevel.PRIVATE)
public class NonGradientOptimizer implements Optimizer {

    private final ProbabilisticModel probabilisticModel;

    /**
     * maxEvaluations the maximum number of objective function evaluations before throwing an exception
     * indicating convergence failure.
     */
    private int maxEvaluations;

    /**
     * bounding box around starting point
     */
    private final double boundsRange;

    /**
     * bounds for each specific continuous latent vertex
     */
    private final OptimizerBounds optimizerBounds;

    /**
     * radius around region to start testing points
     */
    private final double initialTrustRegionRadius;

    /**
     * stopping trust region radius
     */
    private final double stoppingTrustRegionRadius;

    private final List<BiConsumer<double[], Double>> onFitnessCalculations = new ArrayList<>();

    public static NonGradientOptimizerBuilder builder() {
        return new NonGradientOptimizerBuilder();
    }

    @Override
    public void addFitnessCalculationHandler(BiConsumer<double[], Double> fitnessCalculationHandler) {
        this.onFitnessCalculations.add(fitnessCalculationHandler);
    }

    @Override
    public void removeFitnessCalculationHandler(BiConsumer<double[], Double> fitnessCalculationHandler) {
        this.onFitnessCalculations.remove(fitnessCalculationHandler);
    }

    private void handleFitnessCalculation(double[] point, Double fitness) {
        for (BiConsumer<double[], Double> fitnessCalculationHandler : onFitnessCalculations) {
            fitnessCalculationHandler.accept(point, fitness);
        }
    }

    private double optimize(FitnessFunction fitnessFunction) {

        StatusBar statusBar = Optimizer.createFitnessStatusBar(this);

        double logProb = probabilisticModel.logProb();

        if (ProbabilityCalculator.isImpossibleLogProb(logProb)) {
            throw new IllegalArgumentException("Cannot start optimizer on zero probability network");
        }

        List<long[]> shapes = probabilisticModel.getLatentVariables().stream().map(Variable::getShape).collect(Collectors.toList());
        BOBYQAOptimizer optimizer = new BOBYQAOptimizer(
            getNumInterpolationPoints(shapes),
            initialTrustRegionRadius,
            stoppingTrustRegionRadius
        );

        double[] startPoint = Optimizer.convertToPoint(getAsDoubleTensors(probabilisticModel.getLatentVariables()));

        double initialFitness = fitnessFunction.fitness().value(startPoint);

        if (ProbabilityCalculator.isImpossibleLogProb(initialFitness)) {
            throw new IllegalArgumentException("Cannot start optimizer on zero probability network");
        }

        ApacheMathSimpleBoundsCalculator boundsCalculator = new ApacheMathSimpleBoundsCalculator(boundsRange, optimizerBounds);
        SimpleBounds bounds = boundsCalculator.getBounds(probabilisticModel.getLatentVariables(), startPoint);

        PointValuePair pointValuePair = optimizer.optimize(
            new MaxEval(maxEvaluations),
            new ObjectiveFunction(fitnessFunction.fitness()),
            bounds,
            MAXIMIZE,
            new InitialGuess(startPoint)
        );

        statusBar.finish();
        return pointValuePair.getValue();
    }

    private int getNumInterpolationPoints(List<long[]> latentVariableShapes) {
        return (int) (2 * Optimizer.totalNumberOfLatentDimensions(latentVariableShapes) + 1);
    }

    @Override
    public double maxAPosteriori() {
        return optimize(new FitnessFunction(
            probabilisticModel,
            false,
            this::handleFitnessCalculation
        ));
    }

    @Override
    public double maxLikelihood() {
        return optimize(new FitnessFunction(
            probabilisticModel,
            true,
            this::handleFitnessCalculation
        ));
    }

    public static class NonGradientOptimizerBuilder {

        private ProbabilisticModel probabilisticModel;

        private int maxEvaluations = Integer.MAX_VALUE;
        private double boundsRange = Double.POSITIVE_INFINITY;
        private OptimizerBounds optimizerBounds = new OptimizerBounds();
        private double initialTrustRegionRadius = BOBYQAOptimizer.DEFAULT_INITIAL_RADIUS;
        private double stoppingTrustRegionRadius = BOBYQAOptimizer.DEFAULT_STOPPING_RADIUS;

        NonGradientOptimizerBuilder() {
        }


        public NonGradientOptimizerBuilder probabilisticModel(ProbabilisticModel probabilisticModel) {
            this.probabilisticModel = probabilisticModel;
            return this;
        }

        public NonGradientOptimizerBuilder maxEvaluations(int maxEvaluations) {
            this.maxEvaluations = maxEvaluations;
            return this;
        }

        public NonGradientOptimizerBuilder boundsRange(double boundsRange) {
            this.boundsRange = boundsRange;
            return this;
        }

        public NonGradientOptimizerBuilder optimizerBounds(OptimizerBounds optimizerBounds) {
            this.optimizerBounds = optimizerBounds;
            return this;
        }

        public NonGradientOptimizerBuilder initialTrustRegionRadius(double initialTrustRegionRadius) {
            this.initialTrustRegionRadius = initialTrustRegionRadius;
            return this;
        }

        public NonGradientOptimizerBuilder stoppingTrustRegionRadius(double stoppingTrustRegionRadius) {
            this.stoppingTrustRegionRadius = stoppingTrustRegionRadius;
            return this;
        }

        public NonGradientOptimizer build() {
            if (probabilisticModel == null) {
                throw new IllegalStateException("Cannot build optimizer without specifying network to optimize.");
            }
            return new NonGradientOptimizer(
                probabilisticModel,
                maxEvaluations,
                boundsRange,
                optimizerBounds,
                initialTrustRegionRadius,
                stoppingTrustRegionRadius
            );
        }

        public String toString() {
            return "NonGradientOptimizer.NonGradientOptimizerBuilder(probabilisticModel=" + this.probabilisticModel + ", maxEvaluations=" + this.maxEvaluations + ", boundsRange=" + this.boundsRange + ", optimizerBounds=" + this.optimizerBounds + ", initialTrustRegionRadius=" + this.initialTrustRegionRadius + ", stoppingTrustRegionRadius=" + this.stoppingTrustRegionRadius + ")";
        }
    }
}