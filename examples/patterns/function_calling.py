"""Usage: python examples/patterns/function_calling.py CHECKPOINT
Select a function from a closed set, then fill its enum arguments one Choice each; none means use the default."""
import sys

from janus.patterns import LocalEvaluator, select_arguments, select_function

tools = {"plot_price": "Plot the price history of one symbol",
         "rolling_correlation": "Rolling correlation between a symbol and a benchmark"}
enums = {"rolling_correlation": {"symbol": {"NVDA": "Nvidia", "AAPL": "Apple"}, "benchmark": {"SPY": "S&P 500", "QQQ": "Nasdaq 100"},
                                 "window": {"1mo": "One month", "1y": "One year"}},
         "plot_price": {"symbol": {"NVDA": "Nvidia", "AAPL": "Apple"}, "style": {"line": "Line", "candle": "Candlestick"}}}
state = "How correlated has Nvidia been with the S&P over the last month?"
evaluator = LocalEvaluator(sys.argv[1])
picked = select_function(evaluator, state, tools)
print(f"function={picked['function']} confidence={picked['confidence']:.3f}")
if picked["function"] is not None:
    call = select_arguments(evaluator, state, picked["function"], enums[picked["function"]])
    print(f"{call['function']}({', '.join(f'{k}={v!r}' for k, v in call['arguments'].items())}) confidence={call['confidence']:.3f}")
