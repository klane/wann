package com.github.klane.wann.function.input;

import com.github.klane.wann.core.Connection;

import java.util.List;
import java.util.function.ToDoubleFunction;

public interface InputFunction extends ToDoubleFunction<List<Connection>> {}
