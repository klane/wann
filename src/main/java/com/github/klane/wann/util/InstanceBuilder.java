package com.github.klane.wann.util;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Doubles;
import javafx.util.Builder;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

public final class InstanceBuilder implements Builder<Instance> {

    private Instances dataSet;
    private final List<Object> values;

    public InstanceBuilder() {
        this.values = new ArrayList<>();
    }

    public InstanceBuilder dataSet(final Instances dataSet) {
        Preconditions.checkNotNull(dataSet);
        this.dataSet = new Instances(dataSet);
        return this;
    }

    private InstanceBuilder values(final Collection<?> values) {
        Preconditions.checkNotNull(values);
        this.values.addAll(values);
        return this;
    }

    public InstanceBuilder values(final double... values) {
        Preconditions.checkNotNull(values);
        return this.values(Doubles.asList(values));
    }

    public InstanceBuilder values(final String... values) {
        Preconditions.checkNotNull(values);
        return this.values(Arrays.asList(values));
    }

    @Override
    public Instance build() {
        Instance instance = new DenseInstance(this.values.size());
        instance.setDataset(this.dataSet);

        for (int i=0; i<this.values.size(); i++) {
            if (this.values.get(i) instanceof Double) {
                instance.setValue(i, (double) this.values.get(i));
            } else if (this.values.get(i) instanceof String) {
                instance.setValue(i, (String) this.values.get(i));
            } else {
                throw new IllegalArgumentException("Values must be either numeric or a String");
            }
        }

        return instance;
    }
}
