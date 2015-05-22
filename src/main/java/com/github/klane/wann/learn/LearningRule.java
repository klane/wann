package com.github.klane.wann.learn;

import com.github.klane.wann.core.Layer;
import com.github.klane.wann.core.Network;
import com.github.klane.wann.function.error.ErrorFunction;
import com.google.common.base.Preconditions;
import org.bitbucket.klane.weka.WekaUtils;
import weka.core.Instance;
import weka.core.Instances;

import java.util.List;
import java.util.function.BiFunction;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public abstract class LearningRule implements BiFunction<Network, Instances, double[]> {

    private final int numEpochs;
    private final ErrorFunction errorFunction;
    Network network;
    Layer output;

    public LearningRule(final int numEpochs, final ErrorFunction errorFunction) {
        Preconditions.checkArgument(numEpochs > 0, "Number of epochs must be positive");
        Preconditions.checkNotNull(errorFunction);

        this.numEpochs = numEpochs;
        this.errorFunction = errorFunction;
    }

    @Override
    public double[] apply(final Network network, final Instances dataSet) {
        Preconditions.checkNotNull(network);
        Preconditions.checkNotNull(dataSet);

        this.network = network;
        this.output = network.getOutputLayer();
        final List<Instance> trainingSet = WekaUtils.toList(dataSet);

        beforeTraining();

        return IntStream.rangeClosed(1, this.numEpochs).mapToDouble(i ->
                this.errorFunction.applyAsDouble(trainingSet.stream().flatMapToDouble(instance ->
                        DoubleStream.of(this.learnInstance(instance))).toArray())).toArray();
    }

    protected void beforeTraining() {}

    protected abstract double[] learnInstance(final Instance instance);
}
