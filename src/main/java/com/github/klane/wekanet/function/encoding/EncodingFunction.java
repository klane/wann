package com.github.klane.wekanet.function.encoding;

import weka.core.Attribute;
import weka.core.Instance;

import java.util.function.BiFunction;

public interface EncodingFunction extends BiFunction<Attribute, Instance, double[]> {}
