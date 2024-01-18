import argparse
import sys


def main(args):
    arg_type = args.sequence
    numbers = args.length

    if arg_type == "triangular":
        return [n * (n + 1) // 2 for n in range(1, numbers + 1)]

    if arg_type == "square":
        return [n * n for n in range(1, numbers + 1)]

    if arg_type == "factorial":
        res = [1]
        if numbers <= 1:
            return res
        for n in range(2, numbers + 1):
            res.append(res[-1] * n)
        return res

    if arg_type == "fibonacci":
        res = [0, 1]
        if numbers == 1:
            return [0]
        if numbers == 2:
            return res
        for n in range(2, numbers):
            res.append(res[-1] + res[-2])
        return res

    if arg_type == "prime":
        res = [2]
        if numbers <= 1:
            return res
        i = 3
        while len(res) < numbers:
            is_prime = True
            for prime in res:
                if i % prime == 0:
                    is_prime = False
                    break
            if is_prime:
                res.append(i)
            i += 1
        return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", dest="length", type=int)
    parser.add_argument("--sequence", dest="sequence")
    try:
        result = main(parser.parse_args())
        if result == None:
            print("invalid choice", file=sys.stderr)
        else:
            print(result)
    except:
        print("invalid choice", file=sys.stderr)
