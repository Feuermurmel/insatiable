import ast
import contextlib
import importlib.resources
import itertools
import pathlib
import sys
from typing import NamedTuple, Optional, NoReturn, Set, List, Tuple, Union, \
    Any, Iterable

from insatiable.expressions import Expr, var, true, false, solve_expr, ite, \
    ExprSolution, boolean_expr, or_all
from insatiable.util import Hashable


class CompilationError(SyntaxError):
    def __init__(self, message, node: Union[ast.stmt, ast.expr]):
        super().__init__(message)

        self.node = node


class UnsatisfiableError(Exception):
    """
    Raised by `run_module()` when no solution was found to run.
    """


def error(message, node) -> NoReturn:
    assert isinstance(node, (ast.stmt, ast.expr)), node

    raise CompilationError(message, node)


def fail_unhandled_node(node) -> NoReturn:
    error(f'Unhandled node type: {type(node).__name__}', node)


class Shape(Hashable):
    def combine_items(self, condition: Expr, then_item, or_else_item):
        """
        Recursively merge the specified items, taking the value from
        `then_item` in the slices where `condition` is true and the value
        from `or_else_item` otherwise.
        """

        raise NotImplementedError

    def python_value(self, item, solution: ExprSolution):
        """
        Evaluate the value in the slice given by the specified solution and
        return it as a plain Python value.
        """

        raise NotImplementedError

    def boolean_value(self, item) -> Expr:
        """
        Convert the item to its "truthyness" value as an `Expr` instance.
        """

        raise NotImplementedError

    def type_function(self, item) -> 'Function':
        """
        Return the function representing the type of values using this shape.
        """

        raise NotImplementedError


class SingletonShape(Shape):
    """
    Base class for shapes which can only represent a single value and which
    need a separate box for different values.

    The item of a box with this shape will always be None.
    """

    def __init__(self, value):
        self.value = value

    def _hashable_key(self):
        return self.value

    def combine_items(self, condition, then_item, or_else_item):
        return None

    def python_value(self, item, solution):
        return self.value


class BooleanShape(Shape):
    def _hashable_key(self):
        return None

    def combine_items(self, condition, then_item, or_else_item):
        return ite(condition, then_item, or_else_item)

    def python_value(self, item, solution):
        return solution(item)

    def boolean_value(self, item) -> Expr:
        return item

    def type_function(self, item):
        # Need to make sure that we return the exact instance also used for
        # the function defined in builtins.insat. Instances of the same
        # function don't compare equal.
        return _native_functions_by_annotation['__bool__']


_boolean_shape = BooleanShape()


class TupleShape(Shape):
    def __init__(self, len: int):
        self.len = len

    def _hashable_key(self):
        return self.len

    def combine_items(self, condition, then_item, or_else_item):
        return [_if(condition, t, o) for t, o in zip(then_item, or_else_item)]

    def python_value(self, item, solution):
        return tuple(i.evaluate(solution) for i in item)

    def boolean_value(self, item) -> Expr:
        return boolean_expr(self.len)

    def type_function(self, item):
        return _native_functions_by_annotation['__tuple__']


class StringShape(SingletonShape):
    def boolean_value(self, item) -> Expr:
        return boolean_expr(self.value)

    def type_function(self, item):
        return _native_functions_by_annotation['__str__']


class FunctionShape(SingletonShape):
    def boolean_value(self, item) -> Expr:
        return true

    def type_function(self, item):
        return _native_functions_by_annotation['__callable__']


class Box(NamedTuple):
    shape: Shape
    item: Any
    slice: Expr = true


class Value:
    def __init__(self, boxes: List[Box]):
        """
        All slices must be pair-wise disjoint and sum to `true`.
        """

        self.boxes = boxes

    def __repr__(self):
        return f'Value({self.boxes})'

    def evaluate(self, solution: ExprSolution):
        for box in self.boxes:
            if solution(box.slice):
                return box.shape.python_value(box.item, solution)
        else:
            assert False


def _boolean_value(value: Expr):
    return Value([Box(_boolean_shape, value)])


def _string_value(value: str):
    return Value([Box(StringShape(value), None)])


def _function_value(function: 'Function'):
    return Value([Box(FunctionShape(function), None)])


def _tuple_value(items: List[Value]):
    return Value([Box(TupleShape(len(items)), items)])


_none_value = _tuple_value([])


def _if(condition: Expr, then: Value, or_else: Value) -> Value:
    """
    Merge two `Value` instances by taking the values from `then` in the
    slices specified by `condition` and the values from `or_else` in the rest
    of the slices.
    """

    # Easy optimizations.
    if condition == true:
        return then

    if condition == false:
        return or_else

    then_boxes = {i.shape: i for i in then.boxes}
    or_else_boxes = {i.shape: i for i in or_else.boxes}

    def iter_new_boxes():
        # Iterate over all shapes, get the boxes for that shape and possibly
        # merge the items.
        for shape in {*then_boxes, *or_else_boxes}:
            # The items in these dummy boxes are not used.
            _, then_item, then_slice = \
                then_boxes.get(shape, Box(..., ..., false))

            _, or_else_item, or_else_slice = \
                or_else_boxes.get(shape, Box(..., ..., false))

            then_slice &= condition
            or_else_slice /= condition

            if then_slice == false:
                # No box is produced if both slices are now constant false.
                if or_else_slice != false:
                    yield Box(shape, or_else_item, or_else_slice)
            elif or_else_slice == false:
                # TODO: Should we "thin" the items, i.e. remove cases in
                #  nested values that don't overlap with the new slice
                #  anymore?
                yield Box(shape, then_item, then_slice)
            else:
                item = shape.combine_items(condition, then_item, or_else_item)
                slice = ite(condition, then_slice, or_else_slice)

                yield Box(shape, item, slice)

    return Value([*iter_new_boxes()])


def _boolean(value: Value) -> Expr:
    """
    Coerce the specified value to a boolean value. This maps "falsy" values
    (the boolean false constant and the empty tuple) to false and everything
    else to true.
    """

    # The union of all truthy boxes.
    return or_all(
        box.shape.boolean_value(box.item) & box.slice
        for box in value.boxes)


class Variable:
    """
    Represents a variable in a Python scope. The variable may or may not have
    been assigned to in a certain slice.

    `self.value` contains the value of the variable. This only has a defined
    value when `assigned_slice` is true.

    `self.slice` is the slice in which the variable represented by this
    instance has been assigned to. Outside that slice, accessing the variable
    produces an error.
    """

    def __init__(self):
        self.value = _none_value
        self.assigned_slice = false

    def __repr__(self):
        def iter_parts():
            if self.assigned_slice != false:
                yield f'value={self.value}'

            yield f'assigned_slice={self.assigned_slice}'

        parts_str = ' '.join(iter_parts())

        return f'<Variable {parts_str}>'

    def write(self, value: Value, slice: Expr):
        self.value = _if(slice, value, self.value)
        self.assigned_slice |= slice

    def unset(self, slice: Expr):
        self.value = _if(slice, _none_value, self.value)
        self.assigned_slice /= slice


class Scope:
    """
    Represents the local scope of a function or the scope of a module.
    """

    def __init__(self, local_names: Set[str], global_names: Set[str],
                 parent: Optional['Scope']):
        """
        :param local_names:
            The names of variables that are defined in this scope..
        :param global_names:
            Names variables from the global scope which which were made
            visible in this scope using a `global` statement.
        :param parent:
            The parent scope. Referenced variables are looked up in that
            scope if they are not found in the scope.
        """

        self.parent = parent

        global_scope = self

        while global_scope.parent is not None:
            global_scope = global_scope.parent

        self.variables = {
            **{i: Variable() for i in local_names},
            **{i: global_scope.variables[i] for i in global_names}}

    def find_variable(self, name: str) -> Optional[Variable]:
        variable = self.variables.get(name)

        if variable is None and self.parent is not None:
            return self.parent.find_variable(name)

        return variable


def _collect_scope(node, outer_scope: Optional[Scope]) -> Scope:
    names = []
    nonlocal_names = []
    global_names = []

    def collect_assignment_target(target: ast.expr):
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, ast.Tuple):
            for i in target.elts:
                collect_assignment_target(i)
        elif isinstance(target, ast.Starred):
            collect_assignment_target(target.value)
        else:
            fail_unhandled_node(target)

    if isinstance(node, ast.FunctionDef):
        names.extend(i.arg for i in node.args.args)

    def walk_block(stmts):
        for stmt in stmts:
            if isinstance(stmt, ast.Import):
                assert all('.' not in i.name for i in stmt.names)

                names.extend(i.asname or i.name for i in stmt.names)
            elif isinstance(stmt, ast.ImportFrom):
                assert all('.' not in i.name for i in stmt.names)
                assert stmt.level == 0

                names.extend(i.asname or i.name for i in stmt.names)
            elif isinstance(stmt, ast.Nonlocal):
                for i in stmt.names:
                    # outer_scope will not be None because we can't have a
                    # nonlocal statement on the module level.
                    if outer_scope.find_variable(i) is None:
                        error(f'Variable {i} not found in any outer scope.',
                              stmt)

                nonlocal_names.extend(stmt.names)
            elif isinstance(stmt, ast.Global):
                # outer_scope will not be None because we can't have a global
                # statement on the module level.
                global_scope = outer_scope

                while global_scope.parent is not None:
                    global_scope = global_scope.parent

                for i in stmt.names:
                    if i not in global_scope.variables:
                        error(f'Variable {i} is never assigned on the module '
                              f'level. Accessing it is not supported.', stmt)

                global_names.extend(stmt.names)
            elif isinstance(stmt, ast.Assign):
                for i in stmt.targets:
                    collect_assignment_target(i)
            elif isinstance(stmt, (ast.If, ast.While, ast.For)):
                walk_block(stmt.body)
                walk_block(stmt.orelse)
            elif isinstance(stmt, ast.FunctionDef):
                names.append(stmt.name)
            elif isinstance(stmt, (ast.Expr, ast.Return, ast.Assert, ast.Raise, ast.Pass)):
                pass
            else:
                fail_unhandled_node(stmt)

    walk_block(node.body)

    global_names = set(global_names)
    local_names = set(names) - set(nonlocal_names) - global_names

    return Scope(local_names, global_names, outer_scope)


class Module:
    def __init__(self, node: ast.Module):
        self.node = node


def insat_module_from_string(source: str, source_name: str) -> Module:
    root_node = ast.parse(source, source_name)

    # Nodes only know their location within the source file, but not the name
    # of the source file itself. Out of simplicity, we just patch the
    # `SourceFile` instance into each node.
    for i in ast.walk(root_node):
        i.source = source
        i.source_name = source_name

    ast.fix_missing_locations(root_node)

    return Module(root_node)


def load_insat_module(path: pathlib.Path) -> Module:
    return insat_module_from_string(path.read_text(encoding='utf-8'), str(path))


# The idea is basically that anything which can be imported via `from
# insatiable import x` already lives in the module scope under a name of the
# form `__insatiable_x__`.
_builtins_module = insat_module_from_string(
    importlib.resources.read_text(__package__, 'builtins.insat'),
    'builtins.insat')

# Name used by builtins.insat for storing the invariant.
_global_invariants = '__invariants__'


class Function:
    def run(self, args: List[Value], state: 'ExecutionState'):
        """
        Run the body of the function with a new scope and the arguments
        assigned to local variables. Leaves the return value in the return
        variable of the state.
        """

        raise NotImplementedError


class DefFunction(Function):
    """
    Represents an instance of a function declared with the `def` keyword.
    """

    def __init__(self, node: ast.FunctionDef, closure_scope: Scope):
        self.node = node
        self.closure_scope = closure_scope

    def __repr__(self):
        return f'<Function {self.node.name}>'

    def run(self, args: List[Value], state: 'ExecutionState'):
        arg_names = [i.arg for i in self.node.args.args]

        if len(args) != len(arg_names):
            state.set_exception(
                f'Function \'{self.node.name}\' called with {len(args)} '
                f'instead {len(arg_names)} arguments.')
        else:
            scope = _collect_scope(self.node, self.closure_scope)

            with state.with_scope(scope):
                # Write arguments to the self's scope.
                for n, a in zip(arg_names, args):
                    state.write_variable(n, a)

                run_block(self.node.body, state)

            # Handle falling out of the function without a return statement.
            state.set_return(_none_value)


class ThunkFunction(Function):
    """
    Represents a function created from a plain expression in places where the
    expression is only conditionally evaluated (e.g. short-circuit boolean
    operations).
    """

    def __init__(self, node: ast.expr, scope: Scope):
        self.node = node
        self.scope = scope

    def run(self, args: List[Value], state: 'ExecutionState'):
        assert not args

        with state.with_scope(self.scope):
            state.set_return(run_expression(self.node, state))


class BoolNativeFunction(Function):
    def run(self, args: List[Value], state: 'ExecutionState'):
        state.set_return(_boolean_value(_get_unique_var()))

    def __repr__(self):
        return 'bool'


class TupleNativeFunction(Function):
    """
    This function has no purpose except for representing the type if tuple
    values.
    """

    def run(self, args, state):
        state.set_return(_tuple_value(args))

    def __repr__(self):
        return 'tuple'


class StrNativeFunction(Function):
    """
    This function has no purpose except for representing the type if tuple
    values.
    """

    def run(self, args, state):
        # TODO: Implement something here.
        assert False

    def __repr__(self):
        return 'str'


class CallableNativeFunction(Function):
    """
    This function has no purpose except for representing the type if tuple
    values.
    """

    def run(self, args, state):
        # TODO: Implement something here.
        assert False

    def __repr__(self):
        return 'callable'


class TypeNativeFunction(Function):
    def run(self, args, state):
        arg, = args

        for shape, item in state.on_each_box(arg):
            state.set_return(_function_value(shape.type_function(item)))


class PrintNativeFunction(Function):
    def run(self, args, state):
        state.add_print_call(_tuple_value(args))
        state.set_return(_none_value)


class DumbEqFunction(Function):
    def run(self, args, state):
        left, right = args

        for left_shape, left_item in state.on_each_box(left):
            for right_shape, right_item in state.on_each_box(right):
                # We know that we only get shapes here which don't use the
                # boxes item.
                equal_expr = boolean_expr(left_shape == right_shape)

                state.set_return(_boolean_value(equal_expr))


# Some functions' implementations are provided by the runtime. These are
# resolved by having a stub def in builtins.isat with a return type
# annotation which is a string value recognized by the runtime. This dict
# maps from those string values to the implementations.
_native_functions_by_annotation = {
    '__bool__': BoolNativeFunction(),
    '__tuple__': TupleNativeFunction(),
    '__str__': StrNativeFunction(),
    '__callable__': CallableNativeFunction(),
    '__type__': TypeNativeFunction(),
    '__print__': PrintNativeFunction(),
    '__dumb_eq__': DumbEqFunction()}


def _get_native_function_impl(node: ast.FunctionDef):
    """
    Detect the return type annotation used to mark functions implemented by
    the runtime and return the implementation.
    """

    if isinstance(node.returns, ast.Constant):
        value = node.returns.value

        if isinstance(value, str):
            if value.startswith('__') and value.endswith('__'):
                return _native_functions_by_annotation[value]

    return None


_unique_var_counter = itertools.count()


def _get_unique_var():
    return var(f'${next(_unique_var_counter)}')


class ExecutionState:
    """
    State of the execution context while executing a series of statements.

    `slice` and the defined slices of `self.ret_value`, `self.exc_value` are
    always disjoint.
    """

    def __init__(self, initial_scope):
        self.slice = true
        self.scope = initial_scope

        self.return_variable = Variable()
        self.exception_variable = Variable()

        self.print_calls: List[Tuple[Value, Expr]] = []

    def __repr__(self):
        return f'<ExecutionState slice={self.slice}>'

    def set_return(self, value):
        # Set the return value for the whole slice and, because of that,
        # remove the whole slice.
        self.return_variable.write(value, self.slice)
        self.slice = false

    def set_exception(self, *message: Union[Value, str]):
        # TODO: Somehow also capture the location.
        items = [
            _string_value(part) if isinstance(part, str) else
            part for part in message]

        self.exception_variable.write(_tuple_value(items), self.slice)
        self.slice = false

    def clear_exception(self) -> Tuple[Value, Expr]:
        """
        Clear the currently thrown exception and add the slice in which an
        exception was thrown back to this execution state's slice. The
        exception value is returned together with its slice.
        """

        exception_value = self.exception_variable.value
        exception_slice = self.exception_variable.assigned_slice

        # Move the slice back from the variable to the execution state's slice.
        self.slice |= exception_slice
        self.exception_variable.unset(exception_slice)

        return exception_value, exception_slice

    def add_print_call(self, args: Value):
        self.print_calls.append((args, self.slice))

    def read_variable(self, name: str) -> Value:
        variable = self.scope.find_variable(name)

        if variable is None:
            # Basically, we're trying to mimic Python's behavior here. A
            # variable can be read even when it's never written in any of the
            # outer scopes. Here we know that it is never written but handle
            # it as if it wasn't written _yet_.
            variable = Variable()

        # Catch slices in which the variable has not been assigned yet.
        for _ in self.on_condition(~variable.assigned_slice):
            self.set_exception(
                f'Variable \'{name}\' referenced before assignment.')

        # TODO: Maybe we should "trim" the value to the current scope to
        #  prevent useless values from being processed.
        return variable.value

    def write_variable(self, name: str, value: Value):
        # TODO: I think we could optimize this _if() call away in many cases.
        #  If we record in what slice a scope was created, we know that that
        #  scope could only ever contain values with that slice. If we then
        #  set a value for that slice, we know we're overwriting all values.

        # We know for sure that there's a scope containing this variable. The
        # scope is resolved beforehand.
        self.scope.find_variable(name).write(value, self.slice)

    def on_condition(self, condition):
        excluded_slice = self.slice / condition
        self.slice &= condition

        # Only run the wrapped block if the new slice is not definitely empty.
        if self.slice != false:
            yield

        # Add the part of the slice excluded by the condition back to the
        # slice.
        self.slice |= excluded_slice

    def on_each_box(self, value: Value) -> Iterable[Tuple[Shape, Any]]:
        """
        Iterate over all boxes in the specified value, yielding the shape and
        item of each box. While iterating, the slice of this execution state
        is updated to match the boxes' slice.

        This method should always be iterated in it's entirety and only using
        a for, without breaking out of the loop or throwing an exception.
        """

        for box in value.boxes:
            for _ in self.on_condition(box.slice):
                yield box.shape, box.item

    @contextlib.contextmanager
    def with_scope(self, scope):
        saved_scope = self.scope
        self.scope = scope

        yield

        self.scope = saved_scope


def run_call(fn_value: Value, args: List[Value], state: ExecutionState) -> Value:
    # Save the return value from the current stack frame and prepare an empty
    # return value.
    saved_return_variable = state.return_variable
    state.return_variable = Variable()

    # Call all function values within the current slice. This will accumulate
    # the return and exception values in the state.
    for shape, item in state.on_each_box(fn_value):
        if isinstance(shape, FunctionShape):
            shape.value.run(args, state)

    # What is left is the slice where we did not have a function to call.
    state.set_exception('Object is not callable:', fn_value)

    return_variable = state.return_variable

    # Restore the return value and add to the slice what was removed by
    # returning a value.
    state.return_variable = saved_return_variable
    state.slice |= return_variable.assigned_slice

    return return_variable.value


def call_special(name: str, args: List[Value], state: ExecutionState) -> Value:
    return run_call(state.read_variable(name), args, state)


def run_expression(node: ast.expr, state: ExecutionState) -> Value:
    if isinstance(node, ast.Constant):
        if node.value is True:
            return _boolean_value(true)
        elif node.value is False:
            return _boolean_value(false)
        elif isinstance(node.value, str):
            return _string_value(node.value)
        else:
            error(f'Unhandled constant: {node.value}', node)
    elif isinstance(node, ast.Name):
        return state.read_variable(node.id)
    elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            value = run_expression(node.operand, state)

            return call_special('__not__', [value], state)
        else:
            error(f'Unsupported unary operation.', node)
    elif isinstance(node, ast.BoolOp):
        special_name = {ast.And: '__and__', ast.Or: '__or__'}[type(node.op)]
        first_node, *rest = node.values

        value = run_expression(first_node, state)

        for i in rest:
            # A parameter-less function is created and passed to the
            # implementation of the short-circuit operator. This allows the
            # expression to only be evaluated when necessary.
            operand_thunk = _function_value(ThunkFunction(i, state.scope))

            value = call_special(special_name, [value, operand_thunk], state)

        return value
    elif isinstance(node, ast.Compare):
        special_function_names = {
            ast.Eq: '__eq__',
            ast.NotEq: '__ne__',
            ast.Is: '__is__',
            ast.IsNot: '__is_not__'}

        first = run_expression(node.left, state)

        # Get all the functions implementing the comparison operations.
        op_values = [
            state.read_variable(special_function_names[type(i)])
            for i in node.ops]

        # Get thunks for all the compared terms after the first.
        term_thunk_values = [
            _function_value(ThunkFunction(i, state.scope))
            for i in node.comparators]

        # Zip each comparison function with the term is precedes.
        ops_and_term_thunks = _tuple_value(
            [
                _tuple_value([op, term])
                for op, term in zip(op_values, term_thunk_values)])

        # Call a special function which implements evaluating the whole
        # comparison expression, adhering to the correct order and conditions
        # of evaluating each term.
        return call_special('__compare__', [first, ops_and_term_thunks], state)
    elif isinstance(node, ast.Tuple):
        # Value to which all values and unpacked tuples are appended.
        result_value = _tuple_value([])

        for i in node.elts:
            if isinstance(i, ast.Starred):
                value = run_expression(i.value, state)
            else:
                value = _tuple_value([run_expression(i, state)])

            # We want to start off with a tuple here, even though this value
            # will be overwritten in all slices of the execution state. But
            # we can't guarantee that on_each_box() won't iterate over boxes
            # with a slice outside the state's slice.
            new_value = _tuple_value([])

            for _, tuple_item in state.on_each_box(result_value):
                for shape, item in state.on_each_box(value):
                    if not isinstance(shape, TupleShape):
                        state.set_exception(
                            'Can only unpack a tuple, got:', value)
                    else:
                        x = _tuple_value(tuple_item + item)
                        new_value = _if(state.slice, x, new_value)

            result_value = new_value

        return result_value
    elif isinstance(node, ast.Call):
        function_value = run_expression(node.func, state)
        args = [run_expression(i, state) for i in node.args]

        return run_call(function_value, args, state)
    else:
        fail_unhandled_node(node)


def run_assignment(target: ast.expr, value: Value, state: ExecutionState):
    if isinstance(target, ast.Name):
        state.write_variable(target.id, value)
    elif isinstance(target, ast.Tuple):
        starred_items = [
            (i, e)
            for i, e in enumerate(target.elts)
            if isinstance(e, ast.Starred)]

        # Number of non-starred elements.
        target_len = len(target.elts) - len(starred_items)

        for shape, item in state.on_each_box(value):
            if not isinstance(shape, TupleShape):
                state.set_exception('Can only unpack a tuple, got:', value)
            elif starred_items:
                (prefix_end, starred), *rest = starred_items
                starred_len = shape.len - target_len

                if rest:
                    error('Multiple starred expressions.', target)

                if starred_len < 0:
                    state.set_exception(
                        f'Need a tuple with at least {target_len} elements, '
                        f'got:', value)
                else:
                    target_suffix_start = prefix_end + 1
                    value_suffix_start = prefix_end + starred_len

                    target_prefix = target.elts[:prefix_end]
                    value_prefix = item[:prefix_end]

                    for t, v in zip(target_prefix, value_prefix):
                        run_assignment(t, v, state)

                    value_starred = \
                        _tuple_value(item[prefix_end:value_suffix_start])

                    run_assignment(starred.value, value_starred, state)

                    target_suffix = target.elts[target_suffix_start:]
                    item_suffix = item[value_suffix_start:]

                    for t, v in zip(target_suffix, item_suffix):
                        run_assignment(t, v, state)
            else:
                if shape.len != target_len:
                    state.set_exception(
                        f'Need a tuple with exactly {target_len} elements, '
                        f'got:', value)
                else:
                    for element, value in zip(target.elts, item):
                        run_assignment(element, value, state)
    else:
        # Should already have failed in _collect_scope().
        assert False


def run_block(stmts: List[ast.stmt], state: ExecutionState):
    for stmt in stmts:
        if isinstance(stmt, ast.ImportFrom):
            assert stmt.module == 'insatiable'

            for alias in stmt.names:
                local_name = alias.asname or alias.name

                # Anything which can be imported from the `insatiable` module
                # already lives in the module scope under a special name.
                value = state.read_variable(f'__insatiable_{alias.name}__')

                state.write_variable(local_name, value)
        elif isinstance(stmt, ast.Assert):
            if stmt.msg is not None:
                error('A message in an assert statement is not supported.', stmt.msg)

            test_value = run_expression(stmt.test, state)
            call_special('__assert__', [test_value], state)
        elif isinstance(stmt, ast.Raise):
            if stmt.cause is not None:
                error('A cause in a raise statement is not supported.', stmt.cause)

            if stmt.exc is None:
                # Allowing the exception value to be omitted without having
                # an active exception deviates from Python's behavior.
                value = _none_value
            else:
                value = run_expression(stmt.exc, state)

            state.set_exception('Exception raised:', value)
        elif isinstance(stmt, ast.Assign):
            value = run_expression(stmt.value, state)

            for target in stmt.targets:
                run_assignment(target, value, state)
        elif isinstance(stmt, ast.If):
            condition_value = run_expression(stmt.test, state)
            condition = _boolean(condition_value)

            for _ in state.on_condition(condition):
                run_block(stmt.body, state)

            for _ in state.on_condition(~condition):
                run_block(stmt.orelse, state)
        elif isinstance(stmt, ast.FunctionDef):
            # Check whether this is a special function whose implementation
            # is provided by the runtime.
            function = _get_native_function_impl(stmt)

            if function is None:
                # All the stuff we don't support.
                assert not stmt.args.posonlyargs
                assert not stmt.args.vararg
                assert not stmt.args.kwonlyargs
                assert not stmt.args.kw_defaults
                assert not stmt.args.kwarg
                assert not stmt.args.defaults
                assert not stmt.decorator_list

                # In the execution context where the function is instantiated,
                # it has no further conditions. When the variable is assigned,
                # the condition of the current execution context is applied.
                function = DefFunction(stmt, state.scope)

            state.write_variable(stmt.name, _function_value(function))
        elif isinstance(stmt, ast.Expr):
            run_expression(stmt.value, state)
        elif isinstance(stmt, ast.Return):
            state.set_return(run_expression(stmt.value, state))
        elif isinstance(stmt, (ast.Global, ast.Nonlocal, ast.Pass)):
            pass
        else:
            fail_unhandled_node(stmt)


class InsatiableProgram:
    def __init__(self, invariant: Expr, exception: Expr, output: List[Tuple[Value, Expr]]):
        self.invariant = invariant
        self.exception = exception
        self.output = output

    def solve(self, satisfy=None) -> Optional['InsatiableSolution']:
        if satisfy is None:
            satisfy = self.invariant

        solution = solve_expr(satisfy)

        if solution is None:
            return None

        evaluated_print_calls = [
            value.evaluate(solution)
            for value, slice in self.output
            if solution(slice)]

        return InsatiableSolution(
            solution(self.invariant),
            solution(self.exception),
            evaluated_print_calls)


def build_program(module: Module) -> InsatiableProgram:
    try:
        # We glue some code in front of the module to define some builtins.
        merged_module_ast = ast.Module(
            body=_builtins_module.node.body + module.node.body,
            type_ignores=[])

        state = ExecutionState(_collect_scope(merged_module_ast, None))

        run_block(merged_module_ast.body, state)

        # The global variable __invariants__ should be set in all slices.
        invariants = _boolean(state.read_variable(_global_invariants))

        exception_value, exception_slice = state.clear_exception()

        # Print an exception, if one was thrown.
        for _ in state.on_condition(exception_slice):
            state.add_print_call(exception_value)

        print_calls = state.print_calls
    except CompilationError as exception:
        node = exception.node
        line = node.source.splitlines()[node.lineno - 1]

        # We add one because it seems that the column offsets of ast nodes
        # are zero-based while the code formatting a SyntaxError expects a
        # one-based index.
        data = (node.source_name, node.lineno, node.col_offset + 1, line)

        raise SyntaxError(exception, data) from None

    return InsatiableProgram(invariants, exception_slice, print_calls)


class InsatiableSolution:
    def __init__(self, invariant: bool, exception: bool, output: List[Any]):
        self.invariant = invariant
        self.exception = exception
        self.output = output

    def run(self, print_dest=sys.stdout):
        """
        Actually execute the side-effects of the program. This includes
        writing the output from all print() calls to the specified file.
        """

        for items in self.output:
            print(*items, file=print_dest)


def solve_program(module: Module) -> Optional[InsatiableSolution]:
    return build_program(module).solve()


def run_program(module: Module):
    solution = solve_program(module)

    if solution is None:
        raise UnsatisfiableError('No solutions found.')

    solution.run()
