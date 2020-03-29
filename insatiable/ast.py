import ast
import contextlib
import importlib.resources
import itertools
import pathlib
import sys
import traceback
from typing import Optional, NoReturn, Set, List, Tuple, Union, Any

from insatiable.expressions import Expr, var, true, false, solve_expr, or_all
from insatiable.util import log


class CompilationError(SyntaxError):
    def __init__(self, message, node: Union[ast.stmt, ast.expr]):
        super().__init__(message)

        self.node = node


def error(message, node) -> NoReturn:
    assert isinstance(node, (ast.stmt, ast.expr)), node

    raise CompilationError(message, node)


def fail_unhandled_node(node) -> NoReturn:
    error(f'Unhandled node type: {type(node).__name__}', node)


class Value:
    def __init__(
            self,
            boolean_value: Expr,
            boolean_slice: Expr,
            functions: List[Tuple['Function', Expr]]):
        """
        All conditions must be pair-wise disjoint. Each member of
        `boolean_value` must imply `boolean_slice`.
        """

        self.boolean_value = boolean_value
        self.boolean_slice = boolean_slice
        self.functions = functions

    def __repr__(self):
        def iter_parts():
            if self.boolean_slice != false:
                yield f' boolean=({self.boolean_value}, {self.boolean_slice})'

            if self.functions:
                yield f' function={self.functions}'

        parts_str = ''.join(iter_parts())

        return f'<Value{parts_str}>'


def _boolean_value(value: Expr):
    return Value(value, true, [])


def _function_value(function: 'Function'):
    return Value(false, false, [(function, true)])


_none_value = _boolean_value(false)


def _if(condition: Expr, then: Value, or_else: Value) -> Value:
    def _mix_exprs(then_expr, or_else_expr):
        return then_expr & condition | or_else_expr / condition

    boolean_value = _mix_exprs(then.boolean_value, or_else.boolean_value)
    boolean_slice = _mix_exprs(then.boolean_slice, or_else.boolean_slice)

    # TODO: Merge slices of identical values.
    def iter_function_values():
        for value, cond in (then, condition), (or_else, ~condition):
            for fn, slice in value.functions:
                slice &= cond

                if slice != false:
                    yield fn, slice

    return Value(boolean_value, boolean_slice, [*iter_function_values()])


def _boolean(value: Value) -> Expr:
    """
    Map the specified value to a boolean expression, which is true wherever
    the value is "truthy".
    """

    exprs = [value.boolean_value, *(s for _, s in value.functions)]

    return or_all(exprs)


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


class Scope:
    """
    Represents the local scope of a function or the scope of a module.
    """

    def __init__(self, local_names: Set[str], global_names: Set[str], parent: Optional['Scope']):
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
                names.extend(
                    i.id for i in stmt.targets if isinstance(i, ast.Name))
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
    def __init__(self, node: ast.FunctionDef, closure_scope: Scope):
        self.node = node
        self.closure_scope = closure_scope

    def __repr__(self):
        return f'<Function {self.node.name}>'


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

        self.print_calls: List[Tuple[List[Union[Value, str]], Expr]] = []

    def __repr__(self):
        return f'<ExecutionState slice={self.slice}>'

    def set_return(self, value):
        # Set the return value for the whole slice and, because of that,
        # remove the whole slice.
        self.return_variable.write(value, self.slice)
        self.slice = false

    def set_exception(self, value):
        self.exception_variable.write(value, self.slice)
        self.slice = false

    def add_print_call(self, *values: Union[Value, str]):
        self.print_calls.append(([*values], self.slice))

    def read_variable(self, name: str) -> Value:
        variable = self.scope.find_variable(name)

        if variable is None:
            # Basically, we're trying to mimic Python's behavior here. A
            # variable can be read even when it's never written in any of the
            # outer scopes. Here we know that it is never written but handle
            # it as if it wasn't written _yet_.
            variable = Variable()

        # Catch slices in which the variable has not been assigned yet.
        with self.with_condition(~variable.assigned_slice):
            self.add_print_call(
                f'Variable \'{name}\' referenced before assignment.')

            self.set_exception(_none_value)

        return variable.value

    def write_variable(self, name: str, value: Value):
        # TODO: I think we could optimize this _if() call away in many cases.
        #  If we record in what slice a scope was created, we know that that
        #  scope could only ever contain values with that slice. If we then
        #  set a value for that slice, we know we're overwriting all values.

        # We know for sure that there's a scope containing this variable. The
        # scope is resolved beforehand.
        self.scope.find_variable(name).write(value, self.slice)

    @contextlib.contextmanager
    def with_condition(self, condition):
        excluded_slice = self.slice / condition
        self.slice &= condition

        yield

        # Add the part of the slice excluded by the condition back to the
        # slice.
        self.slice |= excluded_slice

    @contextlib.contextmanager
    def with_scope(self, scope):
        saved_scope = self.scope
        self.scope = scope

        yield

        self.scope = saved_scope


def get_function_returns_constant(node: ast.FunctionDef):
    """
    Return the value of the function's returns annotation, if it is a plain
    constant.
    """

    if isinstance(node.returns, ast.Constant):
        return node.returns.value
    else:
        return None


def run_function(function: Function, args: List[Value], state: ExecutionState):
    """
    Runs the body of the function with a new scope and the argument assigned
    to local variables. Leaves the return value in the return variable of the
    state.
    """

    assert not function.node.args.posonlyargs
    # assert not function.node.args.args
    assert not function.node.args.vararg
    assert not function.node.args.kwonlyargs
    assert not function.node.args.kw_defaults
    assert not function.node.args.kwarg
    assert not function.node.args.defaults
    assert not function.node.decorator_list

    returns_constant = get_function_returns_constant(function.node)

    # Check whether this is a special function providing functionality
    # provided by the runtime.
    if returns_constant == '__bool__':
        state.set_return(_boolean_value(_get_unique_var()))
    elif returns_constant == '__print__':
        # TODO: Handle multiple arguments.
        arg, = args

        state.add_print_call(arg)
        state.set_return(_none_value)
    else:
        arg_names = [i.arg for i in function.node.args.args]

        if len(args) != len(arg_names):
            state.add_print_call(
                f'Function \'{function.node.name}\' called with {len(args)} '
                f'instead {len(arg_names)} arguments.')

            state.set_exception(_boolean_value(false))
        else:
            scope = _collect_scope(function.node, function.closure_scope)

            with state.with_scope(scope):
                # Write arguments to the function's scope.
                for n, a in zip(arg_names, args):
                    state.write_variable(n, a)

                run_block(function.node.body, state)

            # Handle falling out of the function without a return statement.
            state.set_return(_none_value)


def run_call(fn_value: Value, args: List[Value], state: ExecutionState) -> Value:
    # Save the return value from the current stack frame and prepare an empty
    # return value.
    saved_return_variable = state.return_variable
    state.return_variable = Variable()

    # Call all function values within the current slice. This will accumulate
    # the return and exception values in the state.
    for fn, fn_slice in fn_value.functions:
        with state.with_condition(fn_slice):
            # Try to optimize some obvious nonsense.
            if state.slice != false:
                run_function(fn, args, state)

    # What is left is the slice where we did not have a function to call.
    state.add_print_call('Object is not callable:', fn_value)
    state.set_exception(_none_value)

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
            right_value = run_expression(i, state)
            value = call_special(special_name, [value, right_value], state)

        return value
    elif isinstance(node, ast.Call):
        function_value = run_expression(node.func, state)
        args = [run_expression(i, state) for i in node.args]

        return run_call(function_value, args, state)
    else:
        fail_unhandled_node(node)


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

            state.add_print_call('Exception raised:', value)
            state.set_exception(value)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    value = run_expression(stmt.value, state)

                    state.write_variable(target.id, value)
                else:
                    fail_unhandled_node(stmt.target)
        elif isinstance(stmt, ast.If):
            condition_value = run_expression(stmt.test, state)
            condition = _boolean(condition_value)

            with state.with_condition(condition):
                run_block(stmt.body, state)

            with state.with_condition(~condition):
                run_block(stmt.orelse, state)
        elif isinstance(stmt, ast.FunctionDef):
            # In the execution context where the function is instantiated,
            # it has no further conditions. When the variable is assigned,
            # the condition of the current execution context is applied.
            value = _function_value(Function(stmt, state.scope))

            state.write_variable(stmt.name, value)
        elif isinstance(stmt, ast.Expr):
            run_expression(stmt.value, state)
        elif isinstance(stmt, ast.Return):
            state.set_return(run_expression(stmt.value, state))
        elif isinstance(stmt, (ast.Global, ast.Nonlocal, ast.Pass)):
            pass
        else:
            fail_unhandled_node(stmt)


class InsatiableSolution:
    def __init__(self, print_calls: List[List[Any]]):
        self.print_calls = print_calls

    def run(self, print_dest=sys.stdout):
        """
        Actually execute the side-effects of the program. This includes
        writing the output from all print() calls to the specified file.
        """

        for items in self.print_calls:
            print(*items, file=print_dest)


def solve_module(module: Module) -> Optional[InsatiableSolution]:
    try:
        # We glue some code in front of the module to define some builtins.
        merged_module_ast = ast.Module(
            body=_builtins_module.node.body + module.node.body,
            type_ignores=[])

        state = ExecutionState(_collect_scope(merged_module_ast, None))

        run_block(merged_module_ast.body, state)

        # The global variable __invariants__ should be set in all slices.
        invariants = _boolean(state.read_variable(_global_invariants))

        exception_variable = state.exception_variable
        print_calls = state.print_calls
    except CompilationError as e:
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb)

        node = e.node
        line = node.source.splitlines()[node.lineno - 1]

        log(f'  File "{node.source_name}", line {node.lineno}')
        log('    ' + line)
        log('    ' + ' ' * node.col_offset + '^')

        log(f'{type(e).__name__}: {e}')
    else:
        solution = solve_expr(invariants)

        if solution is None:
            return None

        def iter_print_calls():
            for items, slice in print_calls:
                if solution(slice):
                    def iter_items():
                        for j in items:
                            if isinstance(j, Value):
                                if solution(j.boolean_slice):
                                    yield solution(j.boolean_value)
                                else:
                                    for f, s in j.functions:
                                        if solution(s):
                                            yield f

                                            # The slices of a value must be
                                            # disjoint, no more slices should
                                            # match.
                                            break
                                    else:
                                        # We should always have a value in
                                        # the slice that print() was called.
                                        assert False
                            else:
                                yield j

                    yield [*iter_items()]

            if solution(exception_variable.assigned_slice):
                yield ['An assertion failed.']

        return InsatiableSolution([*iter_print_calls()])


def run_module(module: Module):
    solution = solve_module(module)

    if solution is None:
        print('No solutions found.')
    else:
        solution.run()
