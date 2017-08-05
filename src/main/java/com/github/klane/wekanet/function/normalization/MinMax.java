package com.github.klane.wekanet.function.normalization;

import java.util.Arrays;
import java.util.DoubleSummaryStatistics;

public class MinMax {

    private final double min;
    private final double max;

    public MinMax(final double[] values) {
        final DoubleSummaryStatistics stats = Arrays.stream(values).summaryStatistics();
        this.min = stats.getMin();
        this.max = stats.getMax();
    }

    /*public Instances normalized(final Instances dataSet, final Attribute attribute) {
        if (attribute.isNominal()) {

        }

        double value;
        double min = dataSet.instance(0).value(attribute);
        double max = min;
        Instances normalized = new Instances(dataSet, dataSet.numInstances());

        for (Instance instance : WekaUtils.toList(dataSet)) {
            value = instance.value(attribute);
            min = Math.min(min, value);
            max = Math.max(max, value);
        }

        for (Instance instance : WekaUtils.toList(dataSet)) {
            instance.setValue(attribute, (instance.value(attribute) - min) / (max - min));
            normalized.add(instance);
        }

        return normalized;
    }*/

    public double[] normalize(final double[] values) {
        return Arrays.stream(values).map(d -> (d - this.min) / (this.max - this.min)).toArray();

        /*final double[] normalized = new double[values.length];
        double min = Math.min(values[0], values[1]);
        double min = Arrays.stream(values).min().getAsDouble();
        double min = Arrays.stream(values).min().getAsDouble();
        double max = Math.max(values[0], values[1]);

        for (int i=2; i<values.length; i++) {
            min = Math.min(min, values[i]);
            max = Math.max(max, values[i]);
        }*/

        /*for (int i=0; i<values.length; i++) {
            normalized[i] = (values[i] - min) / (max - min);
        }

        return normalized;*/
    }
}
