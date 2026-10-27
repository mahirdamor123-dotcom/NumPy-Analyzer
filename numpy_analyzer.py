"""Interactive OOP toolkit for common NumPy array analytics."""
from __future__ import annotations

import numpy as np


class DataAnalytics:
    """Encapsulates numeric arrays and safe operations on them."""

    def __init__(self):
        self.__arrays: list[np.ndarray] = []

    @property
    def arrays(self) -> tuple[np.ndarray, ...]:
        """Return read-only copies so callers cannot mutate stored arrays."""
        return tuple(a.copy() for a in self.__arrays)

    def add_array(self, values, shape=None) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if shape is not None:
            expected = int(np.prod(shape))
            if array.size != expected:
                raise ValueError(f"Shape {tuple(shape)} requires {expected} values; received {array.size}.")
            array = array.reshape(shape)
        if array.ndim not in (1, 2, 3) or array.size == 0:
            raise ValueError("Only non-empty 1D, 2D, and 3D arrays are supported.")
        self.__arrays.append(array.copy())
        return array.copy()

    def get_array(self, index: int) -> np.ndarray:
        try:
            return self.__arrays[index].copy()
        except IndexError as exc:
            raise ValueError(f"Array index {index + 1} does not exist.") from exc

    def __require_same_shape(self, a, b):
        if a.shape != b.shape:
            raise ValueError(f"Arrays must have the same shape; got {a.shape} and {b.shape}.")

    def elementwise(self, a, b, operation: str) -> np.ndarray:
        self.__require_same_shape(a, b)
        operations = {"1": np.add, "2": np.subtract, "3": np.multiply, "4": np.divide}
        if operation not in operations:
            raise ValueError("Choose an operation from 1 to 4.")
        if operation == "4" and np.any(b == 0):
            raise ZeroDivisionError("The second array contains zero; division is undefined.")
        return operations[operation](a, b)

    @staticmethod
    def matrix_product(a, b):
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("Dot/matrix multiplication requires two 2D arrays.")
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Incompatible matrix shapes {a.shape} and {b.shape}.")
        return np.matmul(a, b)

    @classmethod
    def from_text(cls, text: str, shape=None):
        """Convenience constructor for creating an analyzer with one array."""
        values = cls._parse_values(text)
        instance = cls()
        instance.add_array(values, shape)
        return instance

    @staticmethod
    def _parse_values(text: str) -> np.ndarray:
        try:
            values = np.asarray([float(token) for token in text.split()], dtype=float)
        except ValueError as exc:
            raise ValueError("Enter numeric values separated by spaces.") from exc
        if values.size == 0:
            raise ValueError("Enter at least one numeric value.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Values must be finite numbers.")
        return values

    @staticmethod
    def correlation(a, b) -> float:
        x, y = np.ravel(a), np.ravel(b)
        if x.size != y.size or x.size < 2:
            raise ValueError("Correlation needs arrays with the same shape and at least two values.")
        if np.std(x) == 0 or np.std(y) == 0:
            raise ValueError("Correlation is undefined for a constant array.")
        return float(np.corrcoef(x, y)[0, 1])

    def combine(self, a, b, axis=0):
        if a.ndim != b.ndim:
            raise ValueError("Arrays must have the same number of dimensions to combine.")
        if a.ndim == 1:
            return np.concatenate((a, b))
        try:
            return np.concatenate((a, b), axis=axis)
        except (ValueError, np.AxisError) as exc:
            raise ValueError(f"Arrays cannot be combined along axis {axis}: {exc}") from exc


def read_int(prompt, minimum=1):
    while True:
        try:
            value = int(input(prompt))
            if value < minimum:
                print(f"Enter an integer of at least {minimum}.")
                continue
            return value
        except ValueError:
            print("Enter a whole number.")


def choose_array(app: DataAnalytics):
    arrays = app.arrays
    if not arrays:
        print("Create an array first.")
        return None
    print("\nAvailable arrays:")
    for i, array in enumerate(arrays, 1):
        print(f"{i}. shape={array.shape}, dimensions={array.ndim}\n{array}")
    index = read_int("Choose array number: ")
    try:
        return app.get_array(index - 1)
    except ValueError as exc:
        print(exc)
        return None


def create_array(app: DataAnalytics):
    print("1. 1D array\n2. 2D array\n3. 3D array")
    kind = input("Choose dimensions: ").strip()
    if kind not in {"1", "2", "3"}:
        print("Choose 1, 2, or 3.")
        return
    shape = None
    if kind == "1":
        shape = (read_int("Number of elements: "),)
    elif kind == "2":
        shape = (read_int("Rows: "), read_int("Columns: "))
    else:
        shape = (read_int("Depth: "), read_int("Rows: "), read_int("Columns: "))
    count = int(np.prod(shape))
    try:
        values = DataAnalytics._parse_values(input(f"Enter {count} numbers separated by spaces: "))
        array = app.add_array(values, shape)
        print(f"Created array #{len(app.arrays)}:\n{array}")
        if array.ndim > 1:
            print("1. Index one element  2. Slice rows/planes  3. Skip")
            action = input("Choose: ").strip()
            if action == "1":
                indexes = tuple(read_int(f"Index for axis {axis+1} (1 to {size}): ") - 1 for axis, size in enumerate(array.shape))
                if any(i < 0 or i >= size for i, size in zip(indexes, array.shape)):
                    print("Index is outside the array bounds.")
                else: print("Selected value:", array[indexes])
            elif action == "2":
                print("First slice:\n", array[0])
    except ValueError as exc:
        print(exc)


def math_menu(app):
    a = choose_array(app)
    if a is None: return
    b = choose_array(app)
    if b is None: return
    print("1 Add  2 Subtract  3 Multiply element-wise  4 Divide element-wise  5 Dot/matrix product")
    op = input("Operation: ").strip()
    try:
        result = app.matrix_product(a, b) if op == "5" else app.elementwise(a, b, op)
        print("Result:\n", result)
        if input("Save result as another array? [y/N]: ").strip().lower() == "y":
            app.add_array(result)
            print(f"Saved as array #{len(app.arrays)}")
    except (ValueError, ZeroDivisionError) as exc:
        print(f"Cannot calculate: {exc}")


def combine_split_menu(app):
    a = choose_array(app)
    if a is None: return
    print("1 Combine with another array  2 Split into equal sections")
    choice = input("Choose: ").strip()
    try:
        if choice == "1":
            b = choose_array(app)
            if b is None: return
            axis = 0 if a.ndim == 1 else read_int(f"Axis (1 to {a.ndim}): ") - 1
            result = app.combine(a, b, axis)
            print("Combined array:\n", result)
            if input("Save combined result? [y/N]: ").strip().lower() == "y": app.add_array(result)
        elif choice == "2":
            sections = read_int("Number of sections: ")
            axis = 0 if a.ndim == 1 else read_int(f"Axis (1 to {a.ndim}): ") - 1
            result = np.array_split(a, sections, axis=axis)
            for i, part in enumerate(result, 1): print(f"Part {i}:\n{part}")
        else: print("Choose 1 or 2.")
    except (ValueError, np.exceptions.AxisError) as exc: print(f"Cannot combine/split: {exc}")


def search_sort_filter(app):
    a = choose_array(app)
    if a is None: return
    print("1 Search  2 Sort  3 Filter by threshold")
    choice = input("Choose: ").strip()
    try:
        if choice == "1":
            target = float(input("Value to find: "))
            locations = np.argwhere(a == target)
            print(f"Found {len(locations)} match(es) at indexes: {locations.tolist()}" if len(locations) else "Value not found.")
        elif choice == "2":
            direction = input("1 Ascending  2 Descending: ").strip()
            result = np.sort(a, axis=None if a.ndim == 1 else -1)
            if direction == "2": result = np.flip(result, axis=None if a.ndim == 1 else -1)
            print("Sorted (1D globally, otherwise row-wise):\n", result)
        elif choice == "3":
            threshold = float(input("Threshold: "))
            op = input("1 >  2 >=  3 <  4 <= : ").strip()
            funcs = {"1": np.greater, "2": np.greater_equal, "3": np.less, "4": np.less_equal}
            if op not in funcs: print("Unknown comparison."); return
            print("Matching values:", a[funcs[op](a, threshold)])
        else: print("Choose 1, 2, or 3.")
    except ValueError: print("Enter a valid numeric value.")


def statistics_menu(app):
    a = choose_array(app)
    if a is None: return
    print("1 Sum  2 Mean  3 Median  4 Std dev  5 Variance  6 Min/max  7 Percentile  8 Correlation with another array")
    op = input("Choose: ").strip()
    try:
        funcs = {"1": np.sum, "2": np.mean, "3": np.median, "4": np.std, "5": np.var, "6": None, "7": np.percentile}
        if op == "6": print(f"Minimum: {np.min(a)}  Maximum: {np.max(a)}")
        elif op == "7":
            percentile = float(input("Percentile 0–100: "))
            if not 0 <= percentile <= 100: raise ValueError("Percentile must be from 0 to 100.")
            print(f"{percentile:g}th percentile: {funcs[op](a, percentile)}")
        elif op == "8":
            b = choose_array(app)
            if b is not None: print("Correlation:", app.correlation(a, b))
        elif op in funcs: print("Result:", funcs[op](a))
        else: print("Unknown operation.")
    except ValueError as exc: print(exc)


def main():
    app = DataAnalytics()
    print("Welcome to the NumPy Analyzer!")
    while True:
        print("\n==============================\n1. Create a NumPy array\n2. Mathematical operations\n3. Combine or split arrays\n4. Search, sort, or filter arrays\n5. Aggregates and statistics\n6. Exit")
        choice = input("Choose an option: ").strip()
        if choice == "1": create_array(app)
        elif choice == "2": math_menu(app)
        elif choice == "3": combine_split_menu(app)
        elif choice == "4": search_sort_filter(app)
        elif choice == "5": statistics_menu(app)
        elif choice == "6": print("Thank you for using the NumPy Analyzer!"); break
        else: print("Choose a number from 1 to 6.")


if __name__ == "__main__":
    try: main()
    except (KeyboardInterrupt, EOFError): print("\nGoodbye!")
