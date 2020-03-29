from insatiable.expressions import true, false, var, solve_expr


def main():
    def n_of(n, exprs):
        if 0 <= n <= len(exprs):
            if exprs:
                first, *rest = exprs

                return ~first & n_of(n, rest) | first & n_of(n - 1, rest)
            else:
                return true
        else:
            return false

    expr = n_of(3, [var.x1, var.x2, var.x3, var.x4, var.x5])

    print(expr)
    solution = solve_expr(expr)

    if solution is None:
        print('unsatisfiable')
    else:
        for k, v in sorted(solution.values_by_var.items(), key=lambda x: x[0].name):
            print(f'{k.name}: {v}')


main()
