"""Regression coverage for BFCL's current tree-sitter parser API."""
import pytest
from bfcl_eval.model_handler.parser.java_parser import parse_java_function_call
from bfcl_eval.model_handler.parser.js_parser import parse_javascript_function_call


@pytest.mark.parametrize('parse', [parse_java_function_call, parse_javascript_function_call])
def test_named_argument_and_syntax_error(parse):
    assert parse('foo(a=1);') == [{'foo': {'a': '1'}}]
    with pytest.raises(SyntaxError):
        parse('foo(')
