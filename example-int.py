from insatiable.int import int_const, int_equals, int_var, \
    solve_int_expr


def main():
    int_size = 6

    # 8 + x == 21
    expr = int_equals(int_const(8) + int_var('x', int_size), int_const(21))
    print(expr)

    solution = solve_int_expr(expr)

    if solution is None:
        print('unsatisfiable')
    else:
        for k, v in sorted(solution.items(), key=lambda x: x[0].name):
            print(f'{k.name}: {v}')


main()
