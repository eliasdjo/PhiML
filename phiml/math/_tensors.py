import numbers
import warnings
from contextlib import contextmanager
from typing import Union, TypeVar

from dataclasses import dataclass
from typing import Tuple, Callable, List

import numpy
import numpy as np

from ._magic_ops import PhiTreeNodeType, variable_attributes, copy_with, stack, pack_dims, expand
from ._shape import (Shape,
                     CHANNEL_DIM, BATCH_DIM, SPATIAL_DIM, EMPTY_SHAPE,
                     parse_dim_order, shape_stack, merge_shapes, channel, concat_shapes, primal,
                     SUPERSCRIPT, IncompatibleShapes, INSTANCE_DIM, batch, spatial, dual, instance, shape, DimFilter, non_batch, DEBUG_CHECKS)
from ..backend import NoBackendFound, choose_backend, BACKENDS, get_precision, default_backend, convert as convert_, \
    Backend, ComputeDevice, OBJECTS
from ..backend._dtype import DType, combine_types
from .magic import BoundDim, PhiTreeNode, slicing_dict
from .magic import Shapable


class Tensor:
    """
    Abstract base class to represent structured data of one data type.
    This class replaces the native tensor classes `numpy.ndarray`, `torch.Tensor`, `tensorflow.Tensor` or `jax.numpy.ndarray` as the main data container in Φ-ML.

    `Tensor` instances are different from native tensors in two important ways:

    * The dimensions of Tensors have *names* and *types*.
    * Tensors can have non-uniform shapes, meaning that the size of dimensions can vary along other dimensions.

    To check whether a value is a tensor, use `isinstance(value, Tensor)`.

    To construct a Tensor, use `phiml.math.tensor()`, `phiml.math.wrap()` or one of the basic tensor creation functions,
    see https://tum-pbs.github.io/PhiML/Math.html#tensor-creation .

    Tensors are not editable.
    When backed by an editable native tensor, e.g. a `numpy.ndarray`, do not edit the underlying data structure.
    """

    def native(self, order: Union[str, tuple, list, Shape] = None, singleton_for_const=False):
        """
        Returns a native tensor object with the dimensions ordered according to `order`.
        
        Transposes the underlying tensor to match the name order and adds singleton dimensions for new dimension names.
        If a dimension of the tensor is not listed in `order`, a `ValueError` is raised.

        Args:
            order: (Optional) Order of dimension names as comma-separated string, list or `Shape`.
            singleton_for_const: If `True`, dimensions along which values are guaranteed to be constant will not be expanded to their true size but returned as singleton dimensions.

        Returns:
            Native tensor representation, such as PyTorch tensor or NumPy array.

        Raises:
            ValueError if the tensor cannot be transposed to match target_shape
        """
        raise NotImplementedError(self.__class__)

    def numpy(self, order: Union[str, tuple, list, Shape] = None) -> np.ndarray:
        """
        Converts this tensor to a `numpy.ndarray` with dimensions ordered according to `order`.
        
        *Note*: Using this function breaks the autograd chain. The returned tensor is not differentiable.
        To get a differentiable tensor, use `Tensor.native()` instead.
        
        Transposes the underlying tensor to match the name order and adds singleton dimensions for new dimension names.
        If a dimension of the tensor is not listed in `order`, a `ValueError` is raised.

        If this `Tensor` is backed by a NumPy array, a reference to this array may be returned.

        See Also:
            `phiml.math.numpy()`

        Args:
            order: (Optional) Order of dimension names as comma-separated string, list or `Shape`.

        Returns:
            NumPy representation

        Raises:
            ValueError if the tensor cannot be transposed to match target_shape
        """
        native = self.native(order=order)
        return choose_backend(native).numpy(native)

    def __array__(self, dtype=None):  # NumPy conversion
        if self.rank > 1:
            warnings.warn("Automatic conversion of Φ-ML tensors to NumPy can cause problems because the dimension order is not guaranteed.", SyntaxWarning, stacklevel=3)
        return self.numpy(self._shape)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):  # NumPy interface
        if len(inputs) != 2:
            return NotImplemented
        if ufunc.__name__ == 'multiply':
            if inputs[0] is self:
                return self._op2(inputs[1], lambda x, y: x * y, lambda x, y: choose_backend(x, y).mul(x, y), 'mul', '*')
            else:
                return self._op2(inputs[0], lambda x, y: y * x, lambda x, y: choose_backend(x, y).mul(y, x), 'rmul', '*')
        if ufunc.__name__ == 'add':
            if inputs[0] is self:
                return self._op2(inputs[1], lambda x, y: x + y, lambda x, y: choose_backend(x, y).add(x, y), 'add', '+')
            else:
                return self._op2(inputs[0], lambda x, y: y + x, lambda x, y: choose_backend(x, y).add(y, x), 'radd', '+')
        if ufunc.__name__ == 'subtract':
            if inputs[0] is self:
                return self._op2(inputs[1], lambda x, y: x - y, lambda x, y: choose_backend(x, y).sub(x, y), 'add', '-')
            else:
                return self._op2(inputs[0], lambda x, y: y - x, lambda x, y: choose_backend(x, y).sub(y, x), 'rsub', '-')
        if ufunc.__name__ in ['divide', 'true_divide']:
            if inputs[0] is self:
                return self._op2(inputs[1], lambda x, y: x / y, lambda x, y: choose_backend(x, y).div(x, y), 'true_divide', '/')
            else:
                return self._op2(inputs[0], lambda x, y: y / x, lambda x, y: choose_backend(x, y).div(y, x), 'r_true_divide', '/')
        if ufunc.__name__ == 'floor_divide':
            if inputs[0] is self:
                return self._op2(inputs[1], lambda x, y: x // y, lambda x, y: choose_backend(x, y).floordiv(x, y), 'floor_divide', '//')
            else:
                return self._op2(inputs[0], lambda x, y: y // x, lambda x, y: choose_backend(x, y).floordiv(y, x), 'r_floor_divide', '//')
        if ufunc.__name__ == 'remainder':
            if inputs[0] is self:
                return self._op2(inputs[1], lambda x, y: x % y, lambda x, y: choose_backend(x, y).mod(x, y), 'remainder', '%')
            else:
                return self._op2(inputs[0], lambda x, y: y % x, lambda x, y: choose_backend(x, y).mod(y, x), 'r_remainder', '%')
        if ufunc.__name__ == 'power':
            if inputs[0] is self:
                return self._op2(inputs[1], lambda x, y: x ** y, lambda x, y: choose_backend(x, y).pow(x, y), 'power', '**')
            else:
                return self._op2(inputs[0], lambda x, y: y ** x, lambda x, y: choose_backend(x, y).pow(y, x), 'r_power', '**')
        if ufunc.__name__ == 'equal':
            if _EQUALITY_REDUCE[-1] == 'ref':
                return wrap(inputs[0] is inputs[1])
            elif _EQUALITY_REDUCE[-1] == 'shape_and_value':
                if set(inputs[0].shape) != set(inputs[1].shape):
                    return wrap(False)
                from ._ops import close
                return wrap(close(inputs[0], inputs[1], rel_tolerance=0, abs_tolerance=0))
            if inputs[0] is self:
                return self._op2(inputs[1], lambda x, y: x == y, lambda x, y: choose_backend(x, y).equal(x, y), 'equal', '==')
            else:
                return self._op2(inputs[0], lambda x, y: y == x, lambda x, y: choose_backend(x, y).equal(y, x), 'r_equal', '==')
        if ufunc.__name__ == 'not_equal':
            if _EQUALITY_REDUCE[-1] == 'ref':
                return wrap(inputs[0] is not inputs[1])
            elif _EQUALITY_REDUCE[-1] == 'shape_and_value':
                if set(inputs[0].shape) != set(inputs[1].shape):
                    return wrap(True)
                from ._ops import close
                return wrap(not close(inputs[0], inputs[1], rel_tolerance=0, abs_tolerance=0))
            if inputs[0] is self:
                return self._op2(inputs[1], lambda x, y: x != y, lambda x, y: choose_backend(x, y).not_equal(x, y), 'equal', '!=')
            else:
                return self._op2(inputs[0], lambda x, y: y != x, lambda x, y: choose_backend(x, y).not_equal(y, x), 'r_equal', '!=')
        if ufunc.__name__ == 'greater':
            if inputs[0] is self:
                return self._op2(inputs[1], lambda x, y: x > y, lambda x, y: choose_backend(x, y).greater_than(x, y), 'greater', '>')
            else:
                return self._op2(inputs[0], lambda x, y: y > x, lambda x, y: choose_backend(x, y).greater_than(y, x), 'r_greater', '>')
        if ufunc.__name__ == 'greater_equal':
            if inputs[0] is self:
                return self._op2(inputs[1], lambda x, y: x >= y, lambda x, y: choose_backend(x, y).greater_or_equal(x, y), 'greater_equal', '>=')
            else:
                return self._op2(inputs[0], lambda x, y: y >= x, lambda x, y: choose_backend(x, y).greater_or_equal(y, x), 'r_greater_equal', '>=')
        if ufunc.__name__ == 'less':
            if inputs[0] is self:
                return self._op2(inputs[1], lambda x, y: x < y, lambda x, y: choose_backend(x, y).greater_than(y, x), 'less', '<')
            else:
                return self._op2(inputs[0], lambda x, y: y < x, lambda x, y: choose_backend(x, y).greater_than(x, y), 'r_less', '<')
        if ufunc.__name__ == 'less_equal':
            if inputs[0] is self:
                return self._op2(inputs[1], lambda x, y: x <= y, lambda x, y: choose_backend(x, y).greater_or_equal(y, x), 'less_equal', '<=')
            else:
                return self._op2(inputs[0], lambda x, y: y <= x, lambda x, y: choose_backend(x, y).greater_or_equal(x, y), 'r_less_equal', '<=')
        if ufunc.__name__ == 'left_shift':
            if inputs[0] is self:
                return self._op2(inputs[1], lambda x, y: x << y, lambda x, y: choose_backend(x, y).shift_bits_left(x, y), 'left_shift', '<<')
            else:
                return self._op2(inputs[0], lambda x, y: y << x, lambda x, y: choose_backend(x, y).shift_bits_left(y, x), 'r_left_shift', '<<')
        if ufunc.__name__ == 'right_shift':
            if inputs[0] is self:
                return self._op2(inputs[1], lambda x, y: x >> y, lambda x, y: choose_backend(x, y).shift_bits_right(x, y), 'right_shift', '>>')
            else:
                return self._op2(inputs[0], lambda x, y: y >> x, lambda x, y: choose_backend(x, y).shift_bits_right(y, x), 'r_right_shift', '>>')
        raise NotImplementedError(f"NumPy function '{ufunc.__name__}' is not compatible with Φ-ML tensors.")

    @property
    def dtype(self) -> DType:
        """ Data type of the elements of this `Tensor`. """
        raise NotImplementedError()

    @property
    def shape(self) -> Shape:
        """ The `Shape` lists the dimensions with their sizes, names and types. """
        raise NotImplementedError()

    @property
    def default_backend(self) -> Backend:
        from ._ops import choose_backend_t
        return choose_backend_t(self)

    def _with_shape_replaced(self, new_shape: Shape):
        raise NotImplementedError()

    def _with_natives_replaced(self, natives: list):
        """ Replaces all n _natives() of this Tensor with the first n elements of the list and removes them from the list. """
        raise NotImplementedError()

    @property
    def rank(self) -> int:
        """
        Number of explicit dimensions of this `Tensor`. Equal to `tensor.shape.rank`.
        This replaces [`numpy.ndarray.ndim`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ndim.html) /
        [`torch.Tensor.dim`](https://pytorch.org/docs/master/generated/torch.Tensor.dim.html) /
        [`tf.rank()`](https://www.tensorflow.org/api_docs/python/tf/rank) /
        [`jax.numpy.ndim()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndim.html).
        """
        return self.shape.rank

    @property
    def _is_tracer(self) -> bool:
        """
        Tracers store additional internal information.
        They should not be converted to `native()` in intermediate operations.
        
        TensorStack prevents performing the actual stack operation if one of its component tensors is special.
        """
        raise NotImplementedError(self.__class__)

    def _to_dict(self):
        return cached(self)._to_dict()

    def __len__(self):
        return self.shape.volume if self.rank == 1 else NotImplemented

    def __bool__(self):
        assert self.rank == 0, f"Cannot convert tensor with non-empty shape {self.shape} to bool. Use tensor.any or tensor.all instead."
        from ._ops import all_
        if not self.default_backend.supports(Backend.jit_compile):  # NumPy
            return bool(self.native()) if self.rank == 0 else bool(all_(self).native())
        else:
            # __bool__ does not work with TensorFlow tracing.
            # TensorFlow needs to see a tf.Tensor in loop conditions but won't allow bool() invocations.
            # However, this function must always return a Python bool.
            raise AssertionError("To evaluate the boolean value of a Tensor, use 'Tensor.all'.")

    @property
    def all(self):
        """ Whether all values of this `Tensor` are `True` as a native bool. """
        from ._ops import all_, cast
        if self.rank == 0:
            return cast(self, DType(bool)).native()
        else:
            return all_(self, dim=self.shape).native()

    @property
    def any(self):
        """ Whether this `Tensor` contains a `True` value as a native bool. """
        from ._ops import any_, cast
        if self.rank == 0:
            return cast(self, DType(bool)).native()
        else:
            return any_(self, dim=self.shape).native()

    @property
    def mean(self):
        """ Mean value of this `Tensor` as a native scalar. """
        from ._ops import mean
        return mean(self, dim=self.shape).native()

    @property
    def finite_mean(self):
        """ Mean value of all finite values in this `Tensor` as a native scalar. """
        from ._ops import finite_mean
        return finite_mean(self, dim=self.shape).native()

    @property
    def std(self):
        """ Standard deviation of this `Tensor` as a native scalar. """
        from ._ops import std
        return std(self, dim=self.shape).native()

    @property
    def sum(self):
        """ Sum of all values of this `Tensor` as a native scalar. """
        from ._ops import sum_
        return sum_(self, dim=self.shape).native()

    @property
    def finite_sum(self):
        """ Sum of all finite values of this `Tensor` as a native scalar. """
        from ._ops import finite_sum
        return finite_sum(self, dim=self.shape).native()

    @property
    def min(self):
        """ Minimum value of this `Tensor` as a native scalar. """
        from ._ops import min_
        return min_(self, dim=self.shape).native()

    @property
    def finite_min(self):
        """ Minimum finite value of this `Tensor` as a native scalar. """
        from ._ops import finite_min
        return finite_min(self, dim=self.shape).native()

    @property
    def max(self):
        """ Maximum value of this `Tensor` as a native scalar. """
        from ._ops import max_
        return max_(self, dim=self.shape).native()

    @property
    def finite_max(self):
        """ Maximum finite value of this `Tensor` as a native scalar. """
        from ._ops import finite_max
        return finite_max(self, dim=self.shape).native()

    @property
    def real(self) -> 'Tensor':
        """
        Returns the real part of this tensor.

        See Also:
            `phiml.math.real()`
        """
        from ._ops import real
        return real(self)

    @property
    def imag(self) -> 'Tensor':
        """
        Returns the imaginary part of this tensor.
        If this tensor does not store complex numbers, returns a zero tensor with the same shape and dtype as this tensor.

        See Also:
            `phiml.math.imag()`
        """
        from ._ops import imag
        return imag(self)

    @property
    def available(self) -> bool:
        """
        A tensor is available if it stores concrete values and these can currently be read.

        Tracers used inside jit compilation are typically not available.

        See Also:
            `phiml.math.jit_compile()`.
        """
        if self._is_tracer:
            return False
        natives = self._natives()
        natives_available = [choose_backend(native).is_available(native) for native in natives]
        return all(natives_available)

    @property
    def device(self) -> Union[ComputeDevice, None]:
        """
        Returns the `ComputeDevice` that this tensor is allocated on.
        The device belongs to this tensor's `default_backend`.

        See Also:
            `Tensor.default_backend`.
        """
        natives = self._natives()
        if not natives:
            return None
        return self.default_backend.get_device(natives[0])

    def __int__(self):
        return int(self.native()) if self.shape.volume == 1 else NotImplemented

    def __float__(self):
        return float(self.native()) if self.shape.volume == 1 else NotImplemented

    def __complex__(self):
        return complex(self.native()) if self.shape.volume == 1 else NotImplemented

    def __index__(self):
        assert self.shape.volume == 1, f"Only scalar tensors can be converted to index but has shape {self.shape}"
        assert self.dtype.kind == int, f"Only int tensors can be converted to index but dtype is {self.dtype}"
        return int(self.native())

    def __repr__(self):
        return format_tensor(self, PrintOptions())

    def _repr_pretty_(self, printer, cycle):
        printer.text(format_tensor(self, PrintOptions(colors=DEFAULT_COLORS)))

    def __format__(self, format_spec: str):
        if BROADCAST_FORMATTER.values is not None:
            return BROADCAST_FORMATTER.register_formatted(self, format_spec)
        specs = format_spec.split(':')
        layout_ = 'auto'
        for possible_layout in ['summary', 'full', 'row', 'numpy']:
            if possible_layout in specs:
                assert layout_ == 'auto', f"Two layout identifiers encountered in '{format_spec}'"
                layout_ = possible_layout
        include_shape = 'shape' in specs or (False if 'no-shape' in specs else None)
        include_dtype = 'dtype' in specs or (False if 'no-dtype' in specs else None)
        color = 'color' in specs or (False if 'no-color' in specs else None)
        threshold = 8
        float_format = None
        for spec in specs:
            if spec.startswith('threshold='):
                threshold = int(spec[len('threshold='):])
            elif '.' in spec:
                float_format = spec
        result = format_tensor(self, PrintOptions(layout_, float_format, threshold, color, include_shape, include_dtype))
        return result

    def __getitem__(self, item) -> 'Tensor':
        if isinstance(item, Tensor):
            if item.dtype.kind == bool:
                from ._ops import boolean_mask
                return boolean_mask(self, item.shape, item)
            elif item.dtype.kind == int:
                from ._ops import gather
                return gather(self, item)
            else:
                raise AssertionError(f"Index tensor must be of dtype int (gather) or bool (boolean_mask) but got {item}")
        item = slicing_dict(self, item)
        selections = {}
        sliced = self
        for dim, selection in item.items():
            if dim not in self.shape:
                continue
            selection = self.shape.prepare_gather(dim, selection)
            # Either handle slicing directly or add it to the dict
            if isinstance(selection, (tuple, list)):
                from ._magic_ops import stack
                result = [sliced[{dim: i}] for i in selection]
                stack_dim = sliced.shape[dim].after_gather({dim: selection})
                sliced = stack(result, stack_dim)
            elif isinstance(selection, Tensor) and selection.dtype.kind == bool:
                from ._ops import boolean_mask
                sliced = boolean_mask(sliced, dim, selection)
            elif isinstance(selection, Tensor) and selection.dtype.kind == int:
                from ._ops import gather
                sliced = gather(sliced, selection, dims=dim)
            else:
                selections[dim] = selection
        return sliced._getitem(selections) if selections else sliced

    def _getitem(self, selection: dict) -> 'Tensor':
        """
        Slice the tensor along specified dimensions.

        Args:
          selection: dim_name: str -> Union[int, slice]
          selection: dict: 

        Returns:

        """
        raise NotImplementedError()

    def __setitem__(self, key, value):
        raise SyntaxError("Tensors are not editable to preserve the autodiff chain. This feature might be added in the future. To update part of a tensor, use math.where() or math.scatter()")

    def __unstack__(self, dims: Tuple[str, ...]) -> Tuple['Tensor', ...]:  # from phiml.math.magic.Sliceable
        if len(dims) == 1:
            return self._unstack(dims[0])
        else:
            return NotImplemented

    def _unstack(self, dim: str):
        """
        Splits this tensor along the specified dimension.
        The returned tensors have the same dimensions as this tensor save the unstacked dimension.

        Raises an error if the dimension is not part of the `Shape` of this `Tensor`.

        See Also:
            `TensorDim.unstack()`

        Args:
            dim: name of dimension to unstack

        Returns:
            tuple of tensors

        """
        raise NotImplementedError()

    @staticmethod
    def __stack__(values: tuple, dim: Shape, **_kwargs) -> 'Tensor':
        from ._ops import stack_tensors
        return stack_tensors(values, dim)

    def __expand__(self, dims: Shape, **kwargs) -> 'Tensor':
        return expand_tensor(self, dims)

    @staticmethod
    def __concat__(values: tuple, dim: str, **kwargs) -> 'Tensor':
        from ._ops import concat_tensor
        return concat_tensor(values, dim)

    def __replace_dims__(self, dims: Tuple[str, ...], new_dims: Shape, **kwargs) -> 'Tensor':
        from ._magic_ops import rename_dims
        return self._with_shape_replaced(rename_dims(self.shape, dims, new_dims))

    def __unpack_dim__(self, dim: str, unpacked_dims: Shape, **kwargs) -> 'Tensor':
        if self.shape.is_uniform:
            native = self.native(self.shape.names)
            new_shape = self.shape.without(dim)
            i = self.shape.index(dim)
            for d in unpacked_dims:
                new_shape = new_shape._expand(d, pos=i)
                i += 1
            native_reshaped = choose_backend(native).reshape(native, new_shape.sizes)
            return NativeTensor(native_reshaped, new_shape)
        else:
            tensors = self._tensors
            if dim == self._stack_dim.name:
                for udim in unpacked_dims:
                    tensors = [TensorStack(tensors[o::len(tensors)//udim.size], udim) for o in range(len(tensors)//udim.size)]
                assert len(tensors) == 1
                return tensors[0]
            raise NotImplementedError

    def __pack_dims__(self, dims: Tuple[str, ...], packed_dim: Shape, pos: Union[int, None], **kwargs) -> 'Tensor':
        order = self.shape._order_group(dims)
        if self.shape.is_uniform:
            native = self.native(order)
            if pos is None:
                pos = min(self.shape.indices(dims))
            new_shape = self.shape.without(dims)._expand(packed_dim.with_sizes([self.shape.only(dims).volume]), pos)
            native = choose_backend(native).reshape(native, new_shape.sizes)
            return NativeTensor(native, new_shape)
        else:
            from ._ops import concat_tensor
            from ._magic_ops import pack_dims
            value = cached(self)
            assert isinstance(value, TensorStack)
            assert value._stack_dim.name in dims
            inner_packed = [pack_dims(t, dims, packed_dim) for t in value._tensors]
            return concat_tensor(inner_packed, packed_dim.name)

    def __cast__(self, dtype: DType):
        return self._op1(lambda native: choose_backend(native).cast(native, dtype=dtype))

    def dimension(self, name: Union[str, Shape]) -> 'TensorDim':
        """
        Returns a reference to a specific dimension of this tensor.
        This is equivalent to the syntax `tensor.<name>`.

        The dimension need not be part of the `Tensor.shape` in which case its size is 1.

        Args:
            name: dimension name

        Returns:
            `TensorDim` corresponding to a dimension of this tensor
        """
        if isinstance(name, str):
            return TensorDim(self, name)
        elif isinstance(name, Shape):
            return TensorDim(self, name.name)
        else:
            raise ValueError(name)

    def pack(self, dims, packed_dim):
        """ See `pack_dims()` """
        from ._ops import pack_dims
        return pack_dims(self, dims, packed_dim)

    def unpack(self, dim, unpacked_dims):
        """ See `unpack_dim()` """
        from ._ops import unpack_dim
        return unpack_dim(self, dim, unpacked_dims)

    def __getattr__(self, name):
        if name.startswith('__'):  # called by hasattr in magic ops
            raise AttributeError
        if name.startswith('_'):
            raise AttributeError(f"'{type(self)}' object has no attribute '{name}'")
        if name == 'is_tensor_like':  # TensorFlow replaces abs() while tracing and checks for this attribute
            raise AttributeError(f"'{type(self)}' object has no attribute '{name}'")
        assert name not in ('shape', '_shape', 'tensor'), name
        return TensorDim(self, name)

    def __add__(self, other):
        return self._op2(other, lambda x, y: x + y, lambda x, y: choose_backend(x, y).add(x, y), 'add', '+')

    def __radd__(self, other):
        return self._op2(other, lambda x, y: y + x, lambda x, y: choose_backend(x, y).add(y, x), 'radd', '+')

    def __sub__(self, other):
        return self._op2(other, lambda x, y: x - y, lambda x, y: choose_backend(x, y).sub(x, y), 'sub', '-')

    def __rsub__(self, other):
        return self._op2(other, lambda x, y: y - x, lambda x, y: choose_backend(x, y).sub(y, x), 'rsub', '-')

    def __and__(self, other):
        return self._op2(other, lambda x, y: x & y, lambda x, y: choose_backend(x, y).and_(x, y), 'and', '&')

    def __rand__(self, other):
        return self._op2(other, lambda x, y: y & x, lambda x, y: choose_backend(x, y).and_(y, x), 'rand', '&')

    def __or__(self, other):
        return self._op2(other, lambda x, y: x | y, lambda x, y: choose_backend(x, y).or_(x, y), 'or', '|')

    def __ror__(self, other):
        return self._op2(other, lambda x, y: y | x, lambda x, y: choose_backend(x, y).or_(y, x), 'ror', '|')

    def __xor__(self, other):
        return self._op2(other, lambda x, y: x ^ y, lambda x, y: choose_backend(x, y).xor(x, y), 'xor', '^')

    def __rxor__(self, other):
        return self._op2(other, lambda x, y: y ^ x, lambda x, y: choose_backend(x, y).xor(y, x), 'rxor', '^')

    def __mul__(self, other):
        return self._op2(other, lambda x, y: x * y, lambda x, y: choose_backend(x, y).mul(x, y), 'mul', '*')

    def __rmul__(self, other):
        return self._op2(other, lambda x, y: y * x, lambda x, y: choose_backend(x, y).mul(y, x), 'rmul', '*')

    def __truediv__(self, other):
        return self._op2(other, lambda x, y: x / y, lambda x, y: choose_backend(x, y).div(x, y), 'truediv', '/')

    def __rtruediv__(self, other):
        return self._op2(other, lambda x, y: y / x, lambda x, y: choose_backend(x, y).div(y, x), 'rtruediv', '/')

    def __divmod__(self, other):
        return self._op2(other, lambda x, y: divmod(x, y), lambda x, y: divmod(x, y), 'divmod', 'divmod')

    def __rdivmod__(self, other):
        return self._op2(other, lambda x, y: divmod(y, x), lambda x, y: divmod(y, x), 'rdivmod', 'divmod')

    def __floordiv__(self, other):
        return self._op2(other, lambda x, y: x // y, lambda x, y: choose_backend(x, y).floordiv(x, y), 'floordiv', '//')

    def __rfloordiv__(self, other):
        return self._op2(other, lambda x, y: y // x, lambda x, y: choose_backend(x, y).floordiv(y, x), 'rfloordiv', '//')

    def __pow__(self, power, modulo=None):
        assert modulo is None
        return self._op2(power, lambda x, y: x ** y, lambda x, y: choose_backend(x, y).pow(x, y), 'pow', '**')

    def __rpow__(self, other):
        return self._op2(other, lambda x, y: y ** x, lambda x, y: choose_backend(x, y).pow(y, x), 'rpow', '**')

    def __mod__(self, other):
        return self._op2(other, lambda x, y: x % y, lambda x, y: choose_backend(x, y).mod(x, y), 'mod', '%')

    def __rmod__(self, other):
        return self._op2(other, lambda x, y: y % x, lambda x, y: choose_backend(x, y).mod(y, x), 'rmod', '%')

    def __eq__(self, other):
        if _EQUALITY_REDUCE[-1] == 'ref':
            return wrap(self is other)
        elif _EQUALITY_REDUCE[-1] == 'shape_and_value':
            if set(self.shape) != set(other.shape):
                return wrap(False)
            from ._ops import close
            return wrap(close(self, other, rel_tolerance=0, abs_tolerance=0))
        if other is None:
            other = float('nan')
        return self._op2(other, lambda x, y: x == y, lambda x, y: choose_backend(x, y).equal(x, y), 'eq', '==')

    def __ne__(self, other):
        if _EQUALITY_REDUCE[-1] == 'ref':
            return wrap(self is not other)
        elif _EQUALITY_REDUCE[-1] == 'shape_and_value':
            if set(self.shape) != set(other.shape):
                return wrap(True)
            from ._ops import close
            return wrap(not close(self, other, rel_tolerance=0, abs_tolerance=0))
        if other is None:
            other = float('nan')
        return self._op2(other, lambda x, y: x != y, lambda x, y: choose_backend(x, y).not_equal(x, y), 'ne', '!=')

    def __lt__(self, other):
        return self._op2(other, lambda x, y: x < y, lambda x, y: choose_backend(x, y).greater_than(y, x), 'lt', '<')

    def __le__(self, other):
        return self._op2(other, lambda x, y: x <= y, lambda x, y: choose_backend(x, y).greater_or_equal(y, x), 'le', '<=')

    def __gt__(self, other):
        return self._op2(other, lambda x, y: x > y, lambda x, y: choose_backend(x, y).greater_than(x, y), 'gt', '>')

    def __ge__(self, other):
        return self._op2(other, lambda x, y: x >= y, lambda x, y: choose_backend(x, y).greater_or_equal(x, y), 'ge', '>=')

    def __lshift__(self, other):
        return self._op2(other, lambda x, y: x << y, lambda x, y: choose_backend(x, y).shift_bits_left(x, y), 'lshift', '<<')

    def __rlshift__(self, other):
        return self._op2(other, lambda y, x: x << y, lambda y, x: choose_backend(x, y).shift_bits_left(x, y), 'lshift', '<<')

    def __rshift__(self, other):
        return self._op2(other, lambda x, y: x >> y, lambda x, y: choose_backend(x, y).shift_bits_right(x, y), 'rshift', '>>')

    def __rrshift__(self, other):
        return self._op2(other, lambda y, x: x >> y, lambda y, x: choose_backend(x, y).shift_bits_right(x, y), 'rshift', '>>')

    def __abs__(self):
        return self._op1(lambda t: choose_backend(t).abs(t))

    def __round__(self, n=None):
        return self._op1(lambda t: choose_backend(t).round(t))

    def __copy__(self):
        return self._op1(lambda t: choose_backend(t).copy(t, only_mutable=True))

    def __deepcopy__(self, memodict={}):
        return self._op1(lambda t: choose_backend(t).copy(t, only_mutable=False))

    def __neg__(self):
        return self._op1(lambda t: -t)

    def __invert__(self):
        return self._op1(lambda t: ~t)

    def __reversed__(self):
        assert self.shape.channel.rank == 1
        return self[::-1]

    def __iter__(self):
        if self.rank == 1:
            return iter(self.native())
        elif self.rank == 0:
            return iter([self.native()])
        else:
            from ._ops import reshaped_native
            native = reshaped_native(self, [self.shape])
            return iter(native)

    def __matmul__(self, other):
        assert isinstance(other, Tensor), f"Matmul '@' requires two Tensor arguments but got {type(other)}"
        match_names = self.shape.dual.as_batch().names
        if not match_names:  # this is not a matrix
            assert self.shape.primal.only(other.shape).is_empty, f"Cannot compute matmul {self.shape} @ {other.shape}. First argument is not a matrix; it has no dual dimensions."
            return self * other
        match_primal = other.shape.only(match_names, reorder=True)
        if not match_primal:
            assert non_batch(other).non_dual.rank == 1, f"Cannot multiply {self.shape} @ {other.shape} because arg2 does not have appropriate non-dual dimensions"
            match_primal = non_batch(other).non_dual
        match_dual = self.shape.dual.only(match_primal.as_dual(), reorder=True)
        left_arg = pack_dims(self, match_dual, dual('_reduce'))
        right_arg = pack_dims(other, match_primal, channel('_reduce'))
        from ._ops import dot
        return dot(left_arg, '~_reduce', right_arg, '_reduce')

    # def __rmatmul__(self, other):

    def _tensor(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            return other
        elif isinstance(other, (tuple, list)) and any(isinstance(v, Tensor) for v in other):
            if 'vector' in self.shape:
                outer_dim = self.shape['vector']
            elif self.shape.channel_rank == 1:
                outer_dim = self.shape.channel
            else:
                raise ValueError(f"Cannot combine tensor of shape {self.shape} with tuple {tuple([type(v).__name__ for v in other])}")
            remaining_shape = self.shape.without(outer_dim)
            other_items = [v if isinstance(v, Tensor) else compatible_tensor(v, compat_shape=remaining_shape, compat_natives=self._natives(), convert=False) for v in other]
            other_stacked = stack(other_items, outer_dim, expand_values=True)
            return other_stacked
        else:
            return compatible_tensor(other, compat_shape=self.shape, compat_natives=self._natives(), convert=False)

    def _op1(self, native_function):
        """
        Transform the values of this tensor given a function that can be applied to any native tensor.

        Args:
          native_function:

        Returns:

        """
        raise NotImplementedError(self.__class__)

    def _op2(self, other, operator: Callable, native_function: Callable, op_name: str = 'unknown', op_symbol: str = '?') -> 'Tensor':
        """
        Apply a broadcast operation on two tensors.

        Args:
            other: second argument
            operator: function (Tensor, Tensor) -> Tensor, used to propagate the operation to children tensors to have Python choose the callee
            native_function: function (native tensor, native tensor) -> native tensor
            op_name: Name of the python function without leading and trailing `__`.
                Examples: 'add', 'radd', 'sub', 'mul', 'and', 'eq', 'ge'.
            op_symbol: Operation symbol, such as '+', '-', '&', '%', '>='

        Returns:
            `Tensor`
        """
        raise NotImplementedError(self.__class__)

    def _natives(self) -> tuple:
        raise NotImplementedError(self.__class__)

    def _spec_dict(self) -> dict:
        raise NotImplementedError(self.__class__)

    @classmethod
    def _from_spec_and_natives(cls, spec: dict, natives: list):
        raise NotImplementedError(cls)

    def _expand(self):
        """ Expands all compressed tensors to their defined size as if they were being used in `Tensor.native()`. """
        warnings.warn("Tensor._expand() is deprecated, use cached(Tensor) instead.", DeprecationWarning)
        raise NotImplementedError(self.__class__)

    def _simplify(self):
        """ Does not cache this value but if it is already cached, returns the cached version. """
        return self


TensorOrTree = TypeVar('TensorOrTree', Tensor, PhiTreeNode, numbers.Number, bool, tuple, list, dict)


class TensorDim(BoundDim):
    """
    Reference to a specific dimension of a `Tensor`.

    To obtain a `TensorDim`, use `Tensor.dimension()` or the syntax `tensor.<dim>`.

    Indexing a `TensorDim` as `tdim[start:stop:step]` returns a sliced `Tensor`.

    See the documentation at https://tum-pbs.github.io/PhiML/Math.html#indexing-slicing-unstacking .
    """

    def __init__(self, tensor: Tensor, name: str):
        super().__init__(tensor, name)
        self.tensor = tensor

    def __len__(self):
        warnings.warn("Use Tensor.dim.size instead of len(Tensor.dim). len() only supports with integer sizes.", DeprecationWarning)
        return self.size

    def as_batch(self, name: str = None):
        """ Returns a shallow copy of the `Tensor` where the type of this dimension is *batch*. """
        return self._as(BATCH_DIM, name)

    def as_spatial(self, name: str = None):
        """ Returns a shallow copy of the `Tensor` where the type of this dimension is *spatial*. """
        return self._as(SPATIAL_DIM, name)

    def as_channel(self, name: str = None):
        """ Returns a shallow copy of the `Tensor` where the type of this dimension is *channel*. """
        return self._as(CHANNEL_DIM, name)

    def as_instance(self, name: str = None):
        """ Returns a shallow copy of the `Tensor` where the type of this dimension is *instance*. """
        return self._as(INSTANCE_DIM, name)

    def as_type(self, dim_type: Union[Callable, str]):
        return self._as(dim_type('d').type if callable(dim_type) else dim_type, None)

    def _as(self, dim_type: str, name: Union[str, None]):
        if not self.exists:
            return self.tensor
        shape = self.tensor.shape
        new_types = list(shape.types)
        new_types[shape.index(self.name)] = dim_type
        new_names = shape.names
        if name is not None:
            new_names = list(new_names)
            new_names[shape.index(self.name)] = name
        new_shape = Shape(shape.sizes, tuple(new_names), tuple(new_types), shape.item_names)
        return self.tensor._with_shape_replaced(new_shape)

    @property
    def index(self):
        return self.tensor.shape.index(self.name)

    def split(self, split_dimensions: Shape):
        """ See `phiml.math.unpack_dim()` """
        warnings.warn("dim.split() is deprecated. Use math.split_dims() instead.", stacklevel=2)
        from ._magic_ops import unpack_dim
        return unpack_dim(self.tensor, self.name, split_dimensions)

    def __mul__(self, other):
        from ._ops import dot
        if isinstance(other, BoundDim):
            return dot(self.obj, (self.name,), other.obj, (other.name,))
        if isinstance(other, (tuple, list)):
            other = wrap(other, self.obj.shape[self.name])
        if isinstance(other, Tensor):
            assert self.name in other.shape, f"Canno reduce '{self.name}' of tensor with shape {self.obj.shape} against tensor with shape {other.shape}. Dimension must be present on both tensors."
            return dot(self.tensor, (self.name,), other, (self.name,))
        else:
            return NotImplemented

    __rmul__ = __matmul__ = __rmatmul__ = __mul__

    def sum(self):
        from ._ops import sum_
        return sum_(self.tensor, self.name)

    def prod(self):
        from ._ops import prod
        return prod(self.tensor, self.name)


_EQUALITY_REDUCE = [None]


@contextmanager
def equality_by_ref():
    """
    Enables Tensor.__bool__
    """
    _EQUALITY_REDUCE.append('ref')
    try:
        yield None
    finally:
        assert _EQUALITY_REDUCE.pop(-1) == 'ref'


@contextmanager
def equality_by_shape_and_value():
    """
    Enables Tensor.__bool__
    """
    _EQUALITY_REDUCE.append('shape_and_value')
    try:
        yield None
    finally:
        assert _EQUALITY_REDUCE.pop(-1) == 'shape_and_value'


class Layout(Tensor):
    """
    Tensor representation of a PyTree consisting of only lists, tuples and leaves.
    Leaves can be any Python object or primitive, including tuples and lists.
    The PyTree may be deeper but only the outer `shape.rank` levels are represented as a tensor.
    """

    def __init__(self, obj, shape: Shape):
        self._obj = obj
        self._shape = shape

    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def dtype(self) -> DType:
        if isinstance(self._obj, bool):
            return DType(bool)
        if isinstance(self._obj, int):
            return DType(int, 64)
        elif isinstance(self._obj, (float, complex)):
            return DType(type(self._obj), precision=64)
        else:
            return DType(object)

    @property
    def default_backend(self):
        return None

    def native(self, order: Union[str, tuple, list, Shape] = None, singleton_for_const=False):
        order = parse_dim_order(order)
        assert order is None or order == self._shape.names, "Layout.native() does not allow for changing the dimension order"
        return self._obj

    def numpy(self, order: Union[str, tuple, list, Shape] = None) -> np.ndarray:
        native = self.native(order=order)
        return numpy.asarray(native)

    def _getitem(self, selection: dict) -> 'Tensor':
        selection_list = [selection.get(dim, None) for dim in self._shape.names]
        native = self._getitem_recursive(self._obj, tuple(selection_list))
        new_shape = self._shape.after_gather(selection)
        return Layout(native, new_shape)

    def __repr__(self):
        return repr(self._obj)

    def __format__(self, format_spec):
        if BROADCAST_FORMATTER.values is not None:
            return BROADCAST_FORMATTER.register_formatted(self, format_spec)
        return repr(self._obj)

    def _unstack(self, dimension: str):
        if dimension == self._shape.names[0]:
            native = tuple(self._obj.values()) if isinstance(self._obj, dict) else self._obj
            inner_shape = self._shape[1:]
            return tuple([Layout(n, inner_shape) for n in native])
        else:
            raise NotImplementedError()

    @staticmethod
    def _getitem_recursive(native, selection: tuple):
        if not selection:
            return native
        native = tuple(native.values()) if isinstance(native, dict) else native
        if len(selection) == 1:
            return native if selection[0] is None else native[selection[0]]
        else:
            if selection[0] is None:
                return type(native)([Layout._getitem_recursive(n, selection[1:]) for n in native])
            if isinstance(selection[0], int):
                return Layout._getitem_recursive(native[selection[0]], selection[1:])
            elif isinstance(selection[0], slice):
                subset = native[selection[0]]
                return type(subset)([Layout._getitem_recursive(n, selection[1:]) for n in subset])
            else:
                raise ValueError(f"Illegal selection: {selection}")

    def _as_list(self):
        return self._as_list_recursive(self._obj, self._shape.rank, [])

    @staticmethod
    def _as_list_recursive(native, dims: int, result: list):
        if dims == 0:
            result.append(native)
        else:
            native = tuple(native.values()) if isinstance(native, dict) else native
            for n in native:
                Layout._as_list_recursive(n, dims - 1, result)
        return result

    @property
    def _is_tracer(self) -> bool:
        return False

    def __bool__(self):
        assert self.rank == 0, f"Cannot convert tensor with non-empty shape {self.shape} to bool. Use tensor.any or tensor.all instead."
        return bool(self._obj)

    def __stack__(self, values: tuple, dim: Shape, **kwargs) -> 'Layout':
        obj = [v.native(self._shape) for v in values]
        new_shape = concat_shapes(dim, self._shape)
        return Layout(obj, new_shape)

    @staticmethod
    def __concat__(values: tuple, dim: str, **kwargs) -> 'Shapable':
        return NotImplemented

    def __flatten__(self, flat_dim: Shape, flatten_batch: bool):
        if not flatten_batch and self._shape.batch:
            raise NotImplementedError
        return layout(self._as_list(), flat_dim)

    def __expand__(self, dims: Shape, **kwargs) -> 'Tensor':
        new_dims = dims.without(self._shape)
        if not new_dims:
            return self
        obj = self._obj
        for dim in reversed(new_dims):
            assert isinstance(dim.size, int), "Can only expand layouts by integer-sized dimensions"
            obj = [obj] * dim.size
        return Layout(obj, concat_shapes(new_dims, self._shape))

    def __replace_dims__(self, dims: Tuple[str, ...], new_dims: Shape, **kwargs) -> 'Tensor':
        new_shape = self._shape.replace(dims, new_dims)
        return Layout(self._obj, new_shape)

    def __pack_dims__(self, dims: Tuple[str, ...], packed_dim: Shape, pos: Union[int, None], **kwargs) -> 'Layout':
        if dims == self.shape.names:
            native = self._as_list()
            return Layout(native, packed_dim.with_size(len(native)))
        else:
            obj = []
            for i in self._shape.only(dims, reorder=True).meshgrid():
                obj.append(self[i].native())
            return Layout(obj, concat_shapes(packed_dim.with_size(self.shape.only(dims).volume), self._shape.without(dims)))

    def __unpack_dim__(self, dim: str, unpacked_dims: Shape, **kwargs) -> 'Layout':
        return NotImplemented

    def __cast__(self, dtype: DType):
        obj = self._recursive_cast(self._obj, self._shape, dtype)
        return Layout(obj, self._shape)

    def __copy__(self):
        return Layout(self._obj, self._shape)

    def __iter__(self):
        if self.rank == 1:
            return iter(self._obj)
        elif self.rank == 0:
            return iter([self._obj])
        else:
            return iter(self._as_list())

    def __eq__(self, other):
        if _EQUALITY_REDUCE[-1]:
            return Tensor.__eq__(self, other)
        return self._op2(other, lambda x, y: x == y, lambda x, y: x == y, 'eq', '==')

    def __ne__(self, other):
        if _EQUALITY_REDUCE[-1]:
            return Tensor.__ne__(self, other)
        return self._op2(other, lambda x, y: x != y, lambda x, y: x != y, 'ne', '!=')
    
    def _assert_close(self, other: Tensor, rel_tolerance: float, abs_tolerance: float, msg: str, verbose: bool):
        from ._ops import assert_close
        inner_test = lambda x, y: assert_close(x, y, rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance, msg=msg, verbose=verbose)
        return self._op2(other, inner_test, inner_test, 'assert_close', '≈')

    def _op2(self, other, operator: Callable, native_function: Callable, op_name: str = 'unknown', op_symbol: str = '?') -> Tensor:
        obj = self._recursive_op2(self._obj, self._shape, other, operator, native_function, op_name)
        new_shape = concat_shapes(self._shape, other.shape.without(self._shape)) if isinstance(other, Tensor) else self._shape
        return Layout(obj, new_shape)

    @staticmethod
    def _recursive_op2(obj, shape: Shape, other, operator, native_function, op_name):
        if shape:
            dim = shape.names[0]
            if isinstance(other, Tensor) and dim in other.shape:
                assert other.shape.get_size(dim) == len(obj), f"Shape mismatch during {op_name}: '{dim}' has size {len(obj)} on layout but {other.shape.get_size(dim)} on other tensor."
                others = [other[{dim: i}] for i in range(len(obj))]
            else:
                others = [other] * len(obj)
            if isinstance(obj, (tuple, list)):
                return type(obj)([Layout._recursive_op2(i, shape[1:], o, operator, native_function, op_name) for i, o in zip(obj, others)])
            elif isinstance(obj, dict):
                return {k: Layout._recursive_op2(v, shape[1:], o, operator, native_function, op_name) for (k, v), o in zip(obj.items(), others)}
        else:  # leaf
            if isinstance(other, Layout) and not other.shape:
                return native_function(obj, other.native())
            if isinstance(other, Tensor):
                return operator(obj, other)
            else:
                return native_function(obj, other)

    def _op1(self, native_function):
        return Layout(self._recursive_op1(self._obj, self._shape, native_function), self._shape)

    @staticmethod
    def _recursive_op1(obj, shape: Shape, native_function):
        raise NotImplementedError
        # if shape:
        #     if isinstance(obj, (tuple, list)):
        #         return type(obj)([Layout._recursive_op1(i, shape[1:], native_function) for i in obj])
        #     else:
        # else:
        #     return native_function(obj)

    @staticmethod
    def _recursive_cast(obj, shape: Shape, dtype: DType):
        if shape:
            if isinstance(obj, (tuple, list)):
                return type(obj)([Layout._recursive_cast(i, shape[1:], dtype) for i in obj])
            elif isinstance(obj, dict):
                return {k: Layout._recursive_cast(v, shape[1:], dtype) for k, v in obj.items()}
            elif isinstance(obj, Tensor):
                assert obj.shape == shape
                from ._ops import cast
                return cast(obj, dtype)
            else:
                raise ValueError(obj)
        else:
            return dtype.kind(obj)


class NativeTensor(Tensor):
    """
    Tensor backed by a (possibly lower-rank) backend-specific tensor.
    The dimension names and types corresponding to the native tensor are stored in _native_shape.
    The property _shape can contain additional dimensions along which the tensor is constant.
    """

    def __init__(self, native_tensor, native_shape: Shape, expanded_shape: Shape = None):
        expanded_shape = native_shape if expanded_shape is None else expanded_shape
        if DEBUG_CHECKS:
            expanded_shape._check_is_valid_tensor_shape()
            backend = choose_backend(native_tensor)
            assert native_shape.is_uniform
            assert expanded_shape.is_uniform
            assert backend.staticshape(native_tensor) == native_shape.sizes, f"Shape {native_shape} does not match native tensor with shape {backend.staticshape(native_tensor)}"
            assert native_shape in expanded_shape
        self._native = native_tensor
        self._shape = expanded_shape
        self._native_shape = native_shape

    def native(self, order: Union[str, tuple, list, Shape] = None, singleton_for_const=False):
        order = parse_dim_order(order, check_rank=self.rank)
        order = self._shape.names if order is None else order
        assert isinstance(order, tuple)  # should not be necessary
        assert all([n in order for n in self._native_shape.names]), f"order must list all essential dimensions but got {order} for tensor {self.shape}"
        backend = self.default_backend
        if order == self._native_shape.names:
            if self.dtype.precision in [None, get_precision()]:
                return self._native
            else:
                return backend.cast(self._native, DType(self.dtype.kind, precision=get_precision()))
        # --- Transpose ---
        perm = self._native_shape.only(order, reorder=False)._perm(self._native_shape.only(order, reorder=True).names)
        if perm != list(range(len(perm))):
            transposed = backend.transpose(self._native, perm)  # this will cast automatically
        else:
            transposed = backend.as_tensor(self._native)
        if len(order) == len(perm):
            return transposed  # nothing to expand
        # --- Expand ---
        slices = [slice(None) if dim in self._native_shape else None for dim in order]
        expanded = transposed[tuple(slices)]
        if not singleton_for_const:
            multiples = [self._shape.get_size(dim) if dim in self._shape and dim not in self._native_shape else 1 for dim in order]
            expanded = backend.tile(expanded, multiples)
        return expanded

    def _cache(self):
        if self._shape == self._native_shape:
            return
        self._native = self.native(order=self._shape)
        self._native_shape = self._shape

    def _cached(self, dims: Shape = None) -> 'NativeTensor':
        if dims is None or self._shape in (dims & self._native_shape):
            return NativeTensor(self.native(order=self._shape), self._shape, self._shape)
        else:
            new_native_shape = dims & self._native_shape
            tmp_tensor = NativeTensor(self._native, self._native_shape, new_native_shape)
            return NativeTensor(tmp_tensor.native(new_native_shape), new_native_shape, self._shape)

    @property
    def collapsed_dims(self):
        return self._shape.without(self._native_shape)

    @property
    def dtype(self):
        return choose_backend(self._native).dtype(self._native)

    @property
    def shape(self):
        return self._shape

    @property
    def default_backend(self) -> Backend:
        return choose_backend(self._native)

    def _with_shape_replaced(self, new_shape):
        if new_shape.rank != self._shape.rank:
            raise IncompatibleShapes(f"Tensor {self} is not compatible with shape {new_shape}", self._shape, new_shape)
        new_shape = Shape(self._shape.sizes, new_shape.names, new_shape.types, new_shape.item_names)
        native_indices = self._shape.indices(self._native_shape)
        new_native_shape = new_shape[native_indices]
        return NativeTensor(self._native, new_native_shape, new_shape)

    def _with_natives_replaced(self, natives: list):
        native = natives.pop(0)
        new_native_shape = self._native_shape.with_sizes(choose_backend(native).shape(native))
        new_shape = self._shape.with_sizes(new_native_shape)
        return NativeTensor(native, new_native_shape, new_shape)

    @property
    def _is_tracer(self) -> bool:
        return False

    def _to_dict(self):
        result = self.shape._to_dict(include_sizes=False)
        if self.rank == 0:
            result['data'] = self.numpy().item()
        else:
            result['data'] = self.numpy(self._shape).tolist()  # works for all 1+ dimensional arrays
        return result

    def _getitem(self, selection: dict):
        if not selection:
            return self
        selections = [slice(None)] * self._native_shape.rank
        for name, sel in selection.items():
            if name in self._native_shape:
                selections[self._native_shape.index(name)] = sel
            elif name not in self._shape:
                assert isinstance(sel, int), f"Attempting slice missing dimension {name} with {selection}"
        gathered = self.default_backend.multi_slice(self._native, tuple(selections)) if selections else self._native
        new_native_shape = self._native_shape.after_gather(selection)
        new_shape = self._shape.after_gather(selection)
        return NativeTensor(gathered, new_native_shape, new_shape)

    def _unstack(self, dim):
        new_shape = self._shape.without(dim)
        new_native_shape = self._native_shape.without(dim)
        if dim in self._native_shape:
            tensors = self.default_backend.unstack(self._native, axis=self._native_shape.index(dim))
            return tuple([NativeTensor(t, new_native_shape, new_shape) for t in tensors])
        else:
            assert dim in self._shape, f"Cannot unstack tensor {self._shape} along non-existant dimension '{dim}'"
            return (NativeTensor(self._native, new_native_shape, new_shape),) * self._shape.get_size(dim)

    def _op1(self, native_function):
        native = native_function(self._native)
        return NativeTensor(native, self._native_shape, self._shape) if native is not None else self

    def _op2(self, other, operator, native_function, op_name: str = 'unknown', op_symbol: str = '?', switch_args=False):
        try:
            other_tensor = self._tensor(other)
            was_converted = not isinstance(other, Tensor)
        except NoBackendFound:
            return NotImplemented
        if not isinstance(other_tensor, NativeTensor) and not was_converted:
            return NotImplemented
        if not isinstance(other_tensor, NativeTensor):
            other_tensor = NativeTensor(other_tensor.native(other_tensor.shape), other_tensor.shape, other_tensor.shape)
        broadcast_shape = self._native_shape & other_tensor._native_shape
        natives = [t.native(order=broadcast_shape, singleton_for_const=True) if t.rank > 0 else t.native() for t in [self, other_tensor]]
        if switch_args:
            natives = natives[::-1]
        result_tensor = native_function(*natives)
        return NativeTensor(result_tensor, broadcast_shape, self._shape & other_tensor._shape)

    def _natives(self) -> tuple:
        return self._native,

    def _spec_dict(self) -> dict:
        return {'type': NativeTensor, 'native_shape': self._native_shape, 'shape': self._shape}

    @classmethod
    def _from_spec_and_natives(cls, spec: dict, natives: list):
        return NativeTensor(natives.pop(0), spec['native_shape'], spec['shape'])

    def _expand(self):
        self._cache()


class TensorStack(Tensor):
    """
    Implicit stack of multiple tensors.
    List of tensors, does not store stacked tensor in memory.

    Args:

    Returns:

    """

    def __init__(self, components: Union[tuple, list], stack_dim: Shape):
        assert isinstance(stack_dim, Shape) and stack_dim.rank == 1, f"stack_dim must be a single-dimension Shape object but got {type(stack_dim)}"
        # assert len(components) > 1, "Use a CollapsedTensor instead"
        for t in components:
            assert isinstance(t, Tensor)
            assert stack_dim.name not in t.shape, f"Cannot stack along '{stack_dim.name}' because the dimension already exists."
        self._tensors = tuple(components)
        self._stack_dim = stack_dim.with_sizes([len(components)], keep_item_names=True)
        try:
            merge_shapes(*self._tensors)
            self._varying_shapes = False
        except IncompatibleShapes:
            self._varying_shapes = True
        self._shape = shape_stack(self._stack_dim, *[t.shape for t in self._tensors])
        self._cached = None

    @property
    def _is_tracer(self) -> bool:
        return any([t._is_tracer for t in self._tensors])

    @property
    def requires_broadcast(self):
        return self._varying_shapes or not self._shape.well_defined or self._is_tracer or self._tensors[0].shape.is_non_uniform
    
    @property
    def stack_dim(self):
        warnings.warn("TensorStack.stack_dim is deprecated. Use Shape.non_uniform instead.", DeprecationWarning, stacklevel=2)
        return self._stack_dim

    def _cache(self):
        if self._cached is None:
            if self.requires_broadcast:
                return None
            elif all([t.shape.is_uniform for t in self._tensors]):
                natives = [t.native(order=self._shape.names) for t in self._tensors]
                native = choose_backend(*natives).concat(natives, axis=self.shape.index(self._stack_dim.name))
                self._cached = NativeTensor(native, self._shape)
            else:  # cache stack_dim on inner tensors
                non_uniform_dim = self._tensors[0].shape.shape.without('dims')
                if len(non_uniform_dim) > 1:
                    raise NotImplementedError
                unstacked = [t._unstack(non_uniform_dim.name) for t in self._tensors]
                stacked = []
                for to_stack in zip(*unstacked):
                    tensor = TensorStack(to_stack, self._stack_dim)._cache()
                    stacked.append(tensor)
                self._cached = TensorStack(stacked, non_uniform_dim)
        return self._cached

    @property
    def dtype(self):
        return combine_types(*[t.dtype for t in self._tensors])

    @property
    def shape(self):
        return self._shape

    def native(self, order: Union[str, tuple, list, Shape] = None, singleton_for_const=False):
        if self._cached is not None:
            return self._cached.native(order=order)
        else:
            order = parse_dim_order(order, check_rank=self.rank)
            # Is only the stack dimension shifted?
            if order is not None and self._shape.without(self._stack_dim).names == tuple(filter(lambda name: name != self._stack_dim.name, order)):
                inner_order = [dim for dim in order if dim != self._stack_dim.name]
                natives = [t.native(inner_order) for t in self._tensors]
                assert self._stack_dim.name in order, f"Dimension {self._stack_dim} missing from 'order'. Got {order} but tensor has shape {self.shape}."
                native = choose_backend(*natives).stack(natives, axis=order.index(self._stack_dim.name))
                return native
            assert not self.shape.is_non_uniform, f"Cannot convert non-uniform tensor with shape {self.shape} to native tensor."
            return self._cache().native(order=order)

    def _with_shape_replaced(self, new_shape: Shape):
        if self._cached is not None:
            return self._cached._with_shape_replaced(new_shape)
        else:
            new_stack_dim = new_shape[self._shape.index(self._stack_dim.name)]
            new_tensors = []
            for t in self._tensors:
                inner_indices = [self.shape.index(d) for d in t.shape.names]
                new_inner_shape = new_shape[inner_indices]
                new_tensors.append(t._with_shape_replaced(new_inner_shape))
            return TensorStack(new_tensors, new_stack_dim)

    def _getitem(self, selection: dict):
        if self._cached is not None:
            return self._cached._getitem(selection)
        if (self._stack_dim.name not in selection or len(selection) != 1) and not self.requires_broadcast:
            return self._cache()._getitem(selection)
        # --- Inner dims ---
        inner_dict = {dim: sel for dim, sel in selection.items() if dim != self._stack_dim.name}
        tensors = self._tensors
        if len(inner_dict) > 0:
            tensors = [t[inner_dict] for t in tensors]
        # --- stack dimension ---
        if self._stack_dim.name in selection:
            selection = selection[self._stack_dim.name]
            if isinstance(selection, int):
                return tensors[selection]
            elif isinstance(selection, slice):
                return TensorStack(tensors[selection], self._stack_dim)
            else:
                raise NotImplementedError(f"{type(selection)} not supported. Only (int, slice) allwoed")
        else:
            return TensorStack(tensors, self._stack_dim)

    def _unstack(self, dim):
        if self._cached is not None:
            return self._cached._unstack(dim)
        if dim == self._stack_dim.name:
            return self._tensors
        else:
            if self.requires_broadcast:
                unstacked = [t._unstack(dim) for t in self._tensors]
                return tuple([TensorStack(items, self._stack_dim) for items in zip(*unstacked)])
            else:
                return self._cache()._unstack(dim)

    def _op1(self, native_function):
        if self.requires_broadcast:
            tensors = [t._op1(native_function) for t in self._tensors]
            return TensorStack(tensors, self._stack_dim)
        else:
            return self._cache()._op1(native_function)

    def _op2(self, other, operator, native_function, op_name: str = 'unknown', op_symbol: str = '?'):
        other = self._tensor(other)
        if self.requires_broadcast:
            if self._stack_dim.name in other.shape:
                other_slices = other._unstack(self._stack_dim.name)
                tensors = [operator(t1, t2) for t1, t2 in zip(self._tensors, other_slices)]
            else:
                tensors = [operator(t, other) for t in self._tensors]
            return TensorStack(tensors, self._stack_dim)
        elif isinstance(other, NativeTensor) or (isinstance(other, TensorStack) and not other.requires_broadcast):
            new_shape, (native1, native2) = broadcastable_native_tensors(self, other)  # ToDo we don't have to expand all
            result_tensor = native_function(native1, native2)
            return NativeTensor(result_tensor, new_shape, new_shape)
        elif isinstance(other, TensorStack) and other.requires_broadcast:
            if other._stack_dim.name in self.shape:
                self_slices = self._unstack(other._stack_dim.name)
                tensors = [operator(t1, t2) for t1, t2 in zip(self_slices, other._tensors)]
            else:
                tensors = [operator(self, t) for t in other._tensors]
            return TensorStack(tensors, self._stack_dim)
        else:
            return NotImplemented

    def _natives(self) -> tuple:
        if self._cached is not None:
            return self._cached._natives()
        else:
            return sum([t._natives() for t in self._tensors], ())

    def _spec_dict(self) -> dict:
        if self._cached is not None:
            return self._cached._spec_dict()
        else:
            return {'type': TensorStack, 'stack_dim': self._stack_dim, 'tensors': [t._spec_dict() for t in self._tensors]}

    @classmethod
    def _from_spec_and_natives(cls, spec: dict, natives: list):
        tensors = [t['type']._from_spec_and_natives(t, natives) for t in spec['tensors']]
        return TensorStack(tensors, spec['stack_dim'])

    def _with_natives_replaced(self, natives: list):
        if self._cached is not None:
            return self._cached._with_natives_replaced(natives)
        else:
            tensors = [t._with_natives_replaced(natives) for t in self._tensors]
            return TensorStack(tensors, self._stack_dim)

    def _expand(self):
        if self.requires_broadcast:
            for t in self._tensors:
                t._expand()
        self._cache()

    @property
    def is_cached(self):
        return self._cached is not None

    def _simplify(self):
        if self.is_cached:
            return self._cached
        else:
            return self


def tensor(data,
           *shape: Shape,
           convert: bool = True,
           default_list_dim=channel('vector')) -> Tensor:  # TODO assume convert_unsupported, add convert_external=False for constants
    """
    Create a Tensor from the specified `data`.
    If `convert=True`, converts `data` to the preferred format of the default backend.

    `data` must be one of the following:
    
    * Number: returns a dimensionless Tensor.
    * Native tensor such as NumPy array, TensorFlow tensor or PyTorch tensor.
    * `tuple` or `list` of numbers: backs the Tensor with native tensor.
    * `tuple` or `list` of non-numbers: creates tensors for the items and stacks them.
    * Tensor: renames dimensions and dimension types if `names` is specified. Converts all internal native values of the tensor if `convert=True`.
    * Shape: creates a 1D tensor listing the dimension sizes.
    
    While specifying `names` is optional in some cases, it is recommended to always specify them.
    
    Dimension types are always inferred from the dimension names if specified.

    Implementations:

    * NumPy: [`numpy.array`](https://numpy.org/doc/stable/reference/generated/numpy.array.html)
    * PyTorch: [`torch.tensor`](https://pytorch.org/docs/stable/generated/torch.tensor.html), [`torch.from_numpy`](https://pytorch.org/docs/stable/generated/torch.from_numpy.html)
    * TensorFlow: [`tf.convert_to_tensor`](https://www.tensorflow.org/api_docs/python/tf/convert_to_tensor)
    * Jax: [`jax.numpy.array`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.array.html)

    See Also:
        `phiml.math.wrap()` which uses `convert=False`, `layout()`.

    Args:
        data: native tensor, scalar, sequence, Shape or Tensor
        shape: Ordered dimensions and types. If sizes are defined, they will be checked against `data`.`
        convert: If True, converts the data to the native format of the current default backend.
            If False, wraps the data in a `Tensor` but keeps the given data reference if possible.

    Raises:
        AssertionError: if dimension names are not provided and cannot automatically be inferred
        ValueError: if `data` is not tensor-like

    Returns:
        Tensor containing same values as data

    Examples:
        >>> tensor([1, 2, 3], channel(vector='x,y,z'))
        (x=1, y=2, z=3)

        >>> tensor([1., 2, 3], channel(vector='x,y,z'))
        (x=1.000, y=2.000, z=3.000) float64

        >>> tensor(numpy.zeros([10, 8, 6, 2]), batch('batch'), spatial('x,y'), channel(vector='x,y'))
        (batchᵇ=10, xˢ=8, yˢ=6, vectorᶜ=x,y) float64 const 0.0

        >>> tensor([(0, 1), (0, 2), (1, 3)], instance('particles'), channel(vector='x,y'))
        (x=0, y=1); (x=0, y=2); (x=1, y=3) (particlesⁱ=3, vectorᶜ=x,y)

        >>> tensor(numpy.random.randn(10))
        (vectorᶜ=10) float64 -0.128 ± 1.197 (-2e+00...2e+00)
    """
    assert all(isinstance(s, Shape) for s in shape), f"Cannot create tensor because shape needs to be one or multiple Shape instances but got {shape}"
    shape = None if len(shape) == 0 else concat_shapes(*shape)
    if isinstance(data, Tensor):
        if convert:
            backend = data.default_backend
            if backend != default_backend():
                data = data._op1(lambda n: convert_(n, use_dlpack=False))
        if shape is None:
            return data
        else:
            if None in shape.sizes:
                shape = shape.with_sizes(data.shape.sizes)
            return data._with_shape_replaced(shape)
    elif isinstance(data, Shape):
        if shape is None:
            shape = channel('dims')
            shape = shape.with_size(data.names)
            data = data.sizes
        elif not shape:
            assert data.rank == 1, f"When wrapping a Shape as a scalar tensor, it must be a rank-1 shape but got {data}"
            data = data.size
        else:
            assert shape.rank == 1, "Can only convert 1D shapes to Tensors"
            shape = shape.with_size(data.names)
            data = data.sizes
    elif isinstance(data, str) or data is None:
        return layout(data)
    elif isinstance(data, (numbers.Number, bool)):
        assert not shape, f"Trying to create a zero-dimensional Tensor from value '{data}' but shape={shape}"
        if convert:
            data = default_backend().as_tensor(data, convert_external=True)
        return NativeTensor(data, EMPTY_SHAPE)
    if isinstance(data, (tuple, list)):
        if all(isinstance(d, (bool, int, float, complex)) for d in data):
            array = np.array(data)
            assert array.dtype != object
            data = array
        elif all(isinstance(d, str) for d in data):
            return layout(data, shape or default_list_dim)
        else:
            try:
                inner_shape = [] if shape is None else [shape[1:]]
                tensors = [d if isinstance(d, Tensor) else tensor(d, *inner_shape, convert=convert) for d in data]
                return stack(tensors, default_list_dim if shape is None else shape[0].with_sizes([len(tensors)]), expand_values=True)
            except IncompatibleShapes:
                assert not convert, f"Cannot convert {data} to tensor given shape {shape}"
                return layout(data, shape or default_list_dim)
            except ValueError:
                assert not convert, f"Cannot convert {data} to tensor"
                return layout(data, shape or default_list_dim)
    try:
        backend = choose_backend(data)
        if shape is None:
            assert backend.ndims(data) <= 1, "Specify dimension names for tensors with more than 1 dimension"
            shape = default_list_dim if backend.ndims(data) == 1 else EMPTY_SHAPE
            shape = shape.with_sizes(backend.staticshape(data))
        else:
            # fill in sizes or check them
            sizes = backend.staticshape(data)
            if len(sizes) != len(shape):
                raise IncompatibleShapes(f"Rank of given shape {shape} does not match data with sizes {sizes}")
            for size, s in zip(sizes, shape.sizes):
                if s is not None:
                    assert s == size, f"Given shape {shape} does not match data with sizes {sizes}. Consider leaving the sizes undefined."
            shape = shape.with_sizes(sizes, keep_item_names=True)
        if convert:
            data = convert_(data, use_dlpack=False)
        return NativeTensor(data, shape)
    except NoBackendFound:
        raise ValueError(f"{type(data)} is not supported. Only (Tensor, tuple, list, np.ndarray, native tensors) are allowed.\nCurrent backends: {BACKENDS}")


def wrap(data,
         *shape: Shape) -> Tensor:
    """ Short for `phiml.math.tensor()` with `convert=False`. """
    return tensor(data, *shape, convert=False)  # TODO inline, simplify


def layout(objects, *shape: Shape) -> Tensor:
    """
    Wraps a Python tree in a `Tensor`, allowing elements to be accessed via dimensions.
    A python tree is a structure of nested `tuple`, `list`, `dict` and *leaf* objects where leaves can be any Python object.

    All keys of `dict` containers must be of type `str`.
    The keys are automatically assigned as item names along that dimension unless conflicting with other elements.

    Strings may also be used as containers.

    Example:
    >>> t = layout({'a': 'text', 'b': [0, 1]}, channel('dict,inner'))
    >>> t.inner[1].dict['a'].native()
    'e'

    See Also:
        `tensor()`, `wrap()`.

    Args:
        objects: PyTree of `list` or `tuple`.
        *shape: Tensor dimensions

    Returns:
        `Tensor`.
        Calling `Tensor.native()` on the returned tensor will return `objects`.
    """
    assert all(isinstance(s, Shape) for s in shape), f"shape needs to be one or multiple Shape instances but got {shape}"
    shape = EMPTY_SHAPE if len(shape) == 0 else concat_shapes(*shape)
    if isinstance(objects, Layout):
        assert objects.shape == shape
        return objects

    if not shape.well_defined:

        def recursive_determine_shape(native, shape: Shape):
            if not shape:
                return shape
            if isinstance(native, dict):
                assert all([isinstance(k, str) for k in native.keys()]), f"All dict keys in PyTrees must be str but got {tuple(native.keys())}"
                shape = shape.replace(shape[0], shape[0].with_size(tuple(native.keys())))
            if shape.rank == 1:
                return shape.with_sizes((len(native),))
            inner_shape = shape[1:]
            if isinstance(native, (tuple, list)):
                inner_shapes = [recursive_determine_shape(n, inner_shape) for n in native]
            elif isinstance(native, dict):
                inner_shapes = [recursive_determine_shape(n, inner_shape) for n in native.values()]
            else:
                raise ValueError(native)
            return shape_stack(shape[0], *inner_shapes)

        shape = recursive_determine_shape(objects, shape)

    return Layout(objects, shape)
    # if shape.volume == 1:
    #     objects = np.asarray(objects, dtype=object)
    #
    # if isinstance(objects, (tuple, list)):
    #     objects = np.asarray(objects, dtype=object)
    # if isinstance(objects, np.ndarray) and objects.dtype == object:
    #     return Layout(objects, shape)
    # else:
    #     assert shape.volume == 1, f"Cannot layout object of type {objects} along {shape}, a tuple, list or object array is required."


def compatible_tensor(data, compat_shape: Shape = None, compat_natives=(), convert=False):
    if isinstance(data, Tensor):
        return data
    elif isinstance(data, Shape):
        if data.spatial.rank == 1:
            return wrap(data.spatial.size)
        assert compat_shape.channel.rank == 1, "Only single-channel tensors support implicit casting from Shape to tensor"
        assert data.rank == compat_shape.channel.volume
        return wrap(data.spatial.sizes, *compat_shape.channel.with_size(data.names))
    else:
        data_type = type(data)
        backend = choose_backend(*compat_natives, data)
        try:
            data = backend.as_tensor(data, convert_external=convert)
            shape = backend.staticshape(data)
        except ValueError as e:
            raise ValueError(e)
        if len(shape) == 0:
            return NativeTensor(data, EMPTY_SHAPE)
        elif isinstance(data, (tuple, list)):  # always channel, add vector if not available
            data = backend.as_tensor(data)
        if len(shape) == compat_shape.channel_rank:
            other_tensor = wrap(data, compat_shape.channel)
            return other_tensor
        if compat_shape.channel_rank > 1 and len(shape) == 1 and 'vector' in compat_shape.channel:
            return wrap(data, compat_shape['vector'].without_sizes())
        elif len(shape) == compat_shape.rank:
            if len(shape) > 1:
                warnings.warn(f"Combining a phiml.math.Tensor with a {data_type} of same shape is not invariant under shape permutations. Please convert the {data_type} to a phiml.math.Tensor first. Shapes: {shape} and {compat_shape}", SyntaxWarning, stacklevel=5)
            return NativeTensor(data, compat_shape.with_sizes(shape))
        else:
            raise ValueError(f"Cannot combine tensor of shape {shape} with tensor of shape {compat_shape}")


def broadcastable_native_tensors(*tensors):
    """
    Expands and transposes the dimensions of the given tensors so that they all have the same dimension order.

    Args:
      *tensors: sequence of Tensors

    Returns:
      shape, native tensors)

    """
    from ._sparse import SparseCoordinateTensor, CompressedSparseMatrix, dense
    if any(isinstance(t, (SparseCoordinateTensor, CompressedSparseMatrix)) for t in tensors) and not all(isinstance(t, (SparseCoordinateTensor, CompressedSparseMatrix)) for t in tensors):
        tensors = [dense(t) for t in tensors]
    broadcast_shape = merge_shapes(*[t.shape for t in tensors])
    natives = [t.native(order=broadcast_shape.names) if t.rank > 0 else t.native() for t in tensors]
    return broadcast_shape, natives


def custom_op2(x: Union[Tensor, float], y: Union[Tensor, float], l_operator, l_native_function, r_operator=None, r_native_function=None, op_name: str = 'unknown', op_symbol: str = None) -> Tensor:
    """
    Perform a custom operator on two tensors.
    This method first tries calling _op2() on the first tensor and if that fails, tries it on the second tensor.

    Args:
      x: Left argument
      y: Right argument
      l_operator: Operator function acting on Tensors
      l_native_function: Operator function acting on natives
      r_operator:  Argument-reversed operator function acting on Tensors
      r_native_function:  Argument-reversed operator function acting on natives
      op_name: Name of the operator function for debugging purposes. Leading 'r' will be added for the operand-reversed version.
      op_symbol: Short name for the operator, independent of argument order.

    Returns:
        `Tensor`
    """
    if op_symbol is None:
        op_symbol = op_name
    x = wrap(x)
    y = wrap(y)
    result = x._op2(y, l_operator, l_native_function, op_name, op_symbol)
    if result is NotImplemented:
        if r_operator is None:
            r_operator = lambda a, b: l_operator(b, a)
        if r_native_function is None:
            r_native_function = lambda a, b: l_native_function(b, a)
        result = y._op2(x, r_operator, r_native_function, f'r{op_name}', op_symbol)
        if result is NotImplemented:
            raise NotImplementedError(f"Operation not supported between {type(x)} and {type(y)}")
    return result


def disassemble_tensors(tensors: Union[Tuple[Tensor, ...], List[Tensor]], expand: bool) -> Tuple[tuple, Tuple[Shape], tuple]:
    """
    Args:
        tensors: Tuple or list of Tensors.
        expand: Whether to add collapsed dimensions to the native tensors.

    Returns:
        natives: tuple of native tensors
        specs: Identification primitives from which the tensor can be reconstructed given the natives.
            One per tensor.
    """
    for t in tensors:
        if isinstance(t, TensorStack) or expand:
            t._expand()
    natives = sum([t._natives() for t in tensors], ())
    shapes = tuple([t.shape for t in tensors])
    specs = tuple([t._spec_dict() for t in tensors])
    return natives, shapes, specs


def assemble_tensors(natives: Union[tuple, list], specs: Union[Tuple[dict, ...], List[dict]]):
    natives = list(natives)
    result = []
    for spec in specs:
        t = spec['type']._from_spec_and_natives(spec, natives)
        result.append(t)
    return result


MISSING_TENSOR = 'missing'
NATIVE_TENSOR = 'native'


def disassemble_tree(obj: PhiTreeNodeType) -> Tuple[PhiTreeNodeType, List[Tensor]]:
    """
    Splits a nested structure of Tensors into the structure without the tensors and an ordered list of tensors.
    Native tensors will be wrapped in phiml.math.Tensors with default dimension names and dimension types `None`.

    See Also:
        `assemble_tree()`

    Args:
        obj: Nested structure of `Tensor` objects.
            Nested structures include: `tuple`, `list`, `dict`, `phiml.math.magic.PhiTreeNode`.

    Returns:
        empty structure: Same structure as `obj` but with the tensors replaced by `None`.
        tensors: Ordered `list` of all contained `Tensor` objects.
    """
    if obj is None:
        return MISSING_TENSOR, []
    elif isinstance(obj, Tensor):
        return None, [obj]
    elif isinstance(obj, (tuple, list)):
        keys = []
        values = []
        for item in obj:
            key, value = disassemble_tree(item)
            keys.append(key)
            values.extend(value)
        return (tuple(keys) if isinstance(obj, tuple) else keys), values
    elif isinstance(obj, dict):
        keys = {}
        values = []
        for name, item in obj.items():
            key, value = disassemble_tree(item)
            keys[name] = key
            values.extend(value)
        return keys, values
    elif isinstance(obj, PhiTreeNode):
        attributes = variable_attributes(obj)
        keys = {}
        values = []
        for attr in attributes:
            key, value = disassemble_tree(getattr(obj, attr))
            keys[attr] = key
            values.extend(value)
        return copy_with(obj, **keys), values
    else:
        try:
            backend = choose_backend(obj)
            if backend == OBJECTS:
                return obj, []
            sizes = backend.staticshape(obj)
            shape = Shape(sizes, tuple([f"dim{i}" for i in range(len(sizes))]), (None,) * len(sizes), (None,) * len(sizes))
            return NATIVE_TENSOR, [NativeTensor(obj, shape)]
        except NoBackendFound:
            return obj, []


def assemble_tree(obj: PhiTreeNodeType, values: List[Tensor]) -> PhiTreeNodeType:
    """ Reverses `disassemble_tree()` given an empty nested structure and a list of tensors. """
    if obj is MISSING_TENSOR:
        return None
    elif obj is NATIVE_TENSOR:
        value = values.pop(0)
        assert isinstance(value, NativeTensor), f"Failed to assemble tree structure. Encountered {value}"
        return value._native
    elif obj is None:
        value = values.pop(0)
        assert isinstance(value, Tensor)
        return value
    elif isinstance(obj, list):
        return [assemble_tree(item, values) for item in obj]
    elif isinstance(obj, tuple):
        return tuple([assemble_tree(item, values) for item in obj])
    elif isinstance(obj, dict):
        return {name: assemble_tree(val, values) for name, val in obj.items()}
    elif isinstance(obj, PhiTreeNode):
        attributes = variable_attributes(obj)
        values = {a: assemble_tree(getattr(obj, a), values) for a in attributes}
        return copy_with(obj, **values)
    else:
        return obj


def cached(t: Union[Tensor, 'PhiTreeNode']) -> Union[Tensor, 'PhiTreeNode']:
    assert isinstance(t, (Tensor, PhiTreeNode)), f"All arguments must be Tensors but got {type(t)}"
    if isinstance(t, NativeTensor):
        return t._cached()
    elif isinstance(t, TensorStack):
        if t._cached is not None:
            return t._cached
        inners = cached(t._tensors)
        if t.requires_broadcast:
            return TensorStack(inners, t._stack_dim)
        else:
            natives = [t.native(order=t.shape.names) for t in inners]
            native = choose_backend(*natives).stack(natives, axis=t.shape.index(t._stack_dim.name))
            return NativeTensor(native, t.shape)
    elif isinstance(t, Layout):
        return t
    elif isinstance(t, PhiTreeNode):
        tree, tensors = disassemble_tree(t)
        tensors_ = [cached(t_) for t_ in tensors]
        return assemble_tree(tree, tensors_)
    else:
        raise AssertionError(f"Cannot cache {type(t)} {t}")


def expand_tensor(value: Tensor, dims: Shape):
    if not dims:
        return value
    dims.assert_all_sizes_defined()
    if isinstance(value, NativeTensor):
        if dims.is_uniform:
            return NativeTensor(value._native, value._native_shape, dims & value._shape)
        else:
            stack_dim = dims.shape.without('dims')
            if stack_dim.rank > 1:
                raise NotImplementedError("Higher-order non-uniform expand() not yet supported")
            unstacked_dims = [dims.after_gather(i) for i in stack_dim.meshgrid()]
            components = [NativeTensor(value._native, value._native_shape, inner_shape) for inner_shape in unstacked_dims]
            return TensorStack(components, stack_dim)
    if isinstance(value, TensorStack):
        expanded = [expand_tensor(v, dims.without(value.stack_dim)) for v in value._tensors]
        return TensorStack(expanded, value.stack_dim)
    raise NotImplementedError


class Dict(dict):
    """
    Dictionary of `Tensor` or `phiml.math.magic.PhiTreeNode` values.
    Dicts are not themselves tensors and do not have a shape.
    Use `layout()` to treat `dict` instances like tensors.

    In addition to dictionary functions, supports mathematical operators with other `Dict`s and lookup via `.key` syntax.
    `Dict` implements `phiml.math.magic.PhiTreeNode` so instances can be passed to math operations like `sin`.
    """

    def __value_attrs__(self):
        return tuple(self.keys())
    
    # --- Dict[key] ---

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)
        
    # --- operators ---
    
    def __neg__(self):
        return Dict({k: -v for k, v in self.items()})
    
    def __invert__(self):
        return Dict({k: ~v for k, v in self.items()})
    
    def __abs__(self):
        return Dict({k: abs(v) for k, v in self.items()})
    
    def __round__(self, n=None):
        return Dict({k: round(v) for k, v in self.items()})

    def __add__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val + other[key] for key, val in self.items()})
        else:
            return Dict({key: val + other for key, val in self.items()})

    def __radd__(self, other):
        if isinstance(other, Dict):
            return Dict({key: other[key] + val for key, val in self.items()})
        else:
            return Dict({key: other + val for key, val in self.items()})

    def __sub__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val - other[key] for key, val in self.items()})
        else:
            return Dict({key: val - other for key, val in self.items()})

    def __rsub__(self, other):
        if isinstance(other, Dict):
            return Dict({key: other[key] - val for key, val in self.items()})
        else:
            return Dict({key: other - val for key, val in self.items()})

    def __mul__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val * other[key] for key, val in self.items()})
        else:
            return Dict({key: val * other for key, val in self.items()})

    def __rmul__(self, other):
        if isinstance(other, Dict):
            return Dict({key: other[key] * val for key, val in self.items()})
        else:
            return Dict({key: other * val for key, val in self.items()})

    def __truediv__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val / other[key] for key, val in self.items()})
        else:
            return Dict({key: val / other for key, val in self.items()})

    def __rtruediv__(self, other):
        if isinstance(other, Dict):
            return Dict({key: other[key] / val for key, val in self.items()})
        else:
            return Dict({key: other / val for key, val in self.items()})

    def __floordiv__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val // other[key] for key, val in self.items()})
        else:
            return Dict({key: val // other for key, val in self.items()})

    def __rfloordiv__(self, other):
        if isinstance(other, Dict):
            return Dict({key: other[key] // val for key, val in self.items()})
        else:
            return Dict({key: other // val for key, val in self.items()})

    def __pow__(self, power, modulo=None):
        assert modulo is None
        if isinstance(power, Dict):
            return Dict({key: val ** power[key] for key, val in self.items()})
        else:
            return Dict({key: val ** power for key, val in self.items()})

    def __rpow__(self, other):
        if isinstance(other, Dict):
            return Dict({key: other[key] ** val for key, val in self.items()})
        else:
            return Dict({key: other ** val for key, val in self.items()})

    def __mod__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val % other[key] for key, val in self.items()})
        else:
            return Dict({key: val % other for key, val in self.items()})

    def __rmod__(self, other):
        if isinstance(other, Dict):
            return Dict({key: other[key] % val for key, val in self.items()})
        else:
            return Dict({key: other % val for key, val in self.items()})

    def __eq__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val == other[key] for key, val in self.items()})
        else:
            return Dict({key: val == other for key, val in self.items()})

    def __ne__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val != other[key] for key, val in self.items()})
        else:
            return Dict({key: val != other for key, val in self.items()})

    def __lt__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val < other[key] for key, val in self.items()})
        else:
            return Dict({key: val < other for key, val in self.items()})

    def __le__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val <= other[key] for key, val in self.items()})
        else:
            return Dict({key: val <= other for key, val in self.items()})

    def __gt__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val > other[key] for key, val in self.items()})
        else:
            return Dict({key: val > other for key, val in self.items()})

    def __ge__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val >= other[key] for key, val in self.items()})
        else:
            return Dict({key: val >= other for key, val in self.items()})

    # --- overridden methods ---

    def copy(self):
        return Dict(self)


def to_dict(value: Union[Tensor, Shape]):
    """
    Returns a serializable form of a `Tensor` or `Shape`.
    The result can be written to a JSON file, for example.

    See Also:
        `from_dict()`.

    Args:
        value: `Tensor` or `Shape`

    Returns:
        Serializable Python tree of primitives
    """
    if isinstance(value, Shape):
        return value._to_dict(include_sizes=True)
    elif isinstance(value, Tensor):
        return value._to_dict()
    raise ValueError(f"Cannot convert {value} to a dict")


def from_dict(dict_: dict, convert=False):
    """
    Loads a `Tensor` or `Shape` from a serialized form.

    See Also:
        `to_dict()`.

    Args:
        dict_: Serialized tensor properties.
        convert: Whether to convert the data to the current backend format or keep it as a Numpy array.

    Returns:
        `Tensor` or `Shape`.
    """
    shape = Shape._from_dict(dict_)
    if 'data' in dict_:
        return tensor(dict_['data'], shape, convert=convert)
    else:
        return shape




class BroadcastFormatter:
    """
    Usage documented in math.__init__.

    How it works:
    * -f calls __neg__ which tells tensors to call register_formatted() instead of formatting normally.
    * Then __sub__ is called which maps the actual string formatting.
    """

    def __init__(self):
        self.values: List[Tensor] = None

    def register_formatted(self, value: Tensor, format_spec: str):
        self.values.append(value)
        return "{" + f"{len(self.values) - 1}:{format_spec}" + "}"

    def format(self, other: str):
        assert isinstance(other, str), "math.f must be used on a string"
        from ._functional import map_
        if self.values is None:
            raise SyntaxError("Use the syntax -f-f\"{tensor}\". Leading '-' is missing.")
        result = map_(other.format, *self.values)
        self.values = None
        return result

    def __sub__(self, other):
        return self.format(other)

    def __neg__(self):
        if self.values is not None:
            raise SyntaxError("-f called twice without formatting string.")
        self.values = []
        return self


BROADCAST_FORMATTER = BroadcastFormatter()


@dataclass
class Color:
    name: str
    console_foreground_begin: str

    def __call__(self, obj, **kwargs):
        text = str(obj).replace(CONSOLE_END, self.console_foreground_begin)
        return f"{self.console_foreground_begin}{text}{CONSOLE_END if self.console_foreground_begin else ''}"


DEFAULT = Color("Default", '')
BLUE = Color("Blue", '\033[94m')
GREEN = Color("Green", '\033[92m')
YELLOW = Color("Yellow", '\033[93m')
GREY = Color("Grey", '\033[37m')
CONSOLE_END = '\033[0m'


@dataclass
class ColorScheme:
    value: Color
    shape: Color
    dtype: Color
    fine: Color


DEFAULT_COLORS = ColorScheme(BLUE, GREEN, YELLOW, GREY)
NO_COLORS = ColorScheme(DEFAULT, DEFAULT, DEFAULT, DEFAULT)


@dataclass
class PrintOptions:
    layout: str = 'auto'
    float_format: str = None
    threshold: int = 8
    colors: ColorScheme = None
    include_shape: bool = None
    include_dtype: bool = None

    def get_colors(self):
        if self.colors is True:
            return DEFAULT_COLORS
        elif self.colors is False:
            return NO_COLORS
        elif self.colors is not None:
            return self.colors
        else:  # None
            return DEFAULT_COLORS if check_is_printing() else NO_COLORS


def check_is_printing():
    import traceback, sys
    stack = traceback.extract_stack()
    for frame in stack:
        if "_pydevd_bundle\\pydevd_xml.py" in frame.filename or "_pydevd_bundle/pydevd_xml.py" in frame.filename:
            return False
    for frame in stack:
        if frame.line.strip().startswith('print('):
            return True
    if 'ipykernel' in sys.modules:
        return True
    return False


def format_summary(self: Tensor, options: PrintOptions) -> str:
    """
    Returns shape + dtype + content summary

    * `bool`: n / N True
    * `float`: mean ± std (min...max)
    """
    if not self.available:
        return format_tracer(self, options)
    from ._sparse import SparseCoordinateTensor, CompressedSparseMatrix
    if isinstance(self, (SparseCoordinateTensor, CompressedSparseMatrix)):
        return sparse_summary(self, options)
    colors = options.get_colors()
    tokens = []
    if self.shape if options.include_shape is None else options.include_shape:
        tokens.append(f"{colors.shape(self.shape)}")
    if is_unexpected_dtype(self.dtype) if options.include_dtype is None else options.include_dtype:
        tokens.append(f"{colors.dtype(self.dtype)}")
    try:
        if self.rank == 0:
            tokens.append(colors.value(self.numpy()))
        elif self.dtype.kind == bool:
            tokens.append(colors.value(f"{self.sum} / {self.shape.volume} True"))
        elif self.dtype.kind in (float, int):
            min_val, max_val, mean, std = [float(f) for f in [self.finite_min, self.finite_max, self.finite_mean, self.std]]
            if std == 0:
                tokens.append(colors.value(f"const {mean:{options.float_format or ''}}"))
            else:
                if any([abs(val) < 0.001 or abs(val) > 1000 for val in [mean, std]]):
                    tokens.append(colors.value(f"{mean:{options.float_format or '.2e'}} ± {std:{options.float_format or '.1e'}}"))
                else:
                    tokens.append(colors.value(f"{mean:{options.float_format or '.3f'}} ± {std:{options.float_format or '.3f'}}"))
                tokens.append(colors.fine(f"({min_val:{options.float_format or '.0e'}}...{max_val:{options.float_format or '.0e'}})"))
        elif self.dtype.kind == complex:
            tokens.append(colors.value(f"|...| < {abs(self).max}"))
    except BaseException as err:
        tokens.append(f"failed to fetch values: {err}")
    return " ".join(tokens)


def sparse_summary(value: Tensor, options: PrintOptions) -> str:
    colors = options.get_colors()
    from ._sparse import get_format
    tokens = []
    if is_unexpected_dtype(value.dtype) if options.include_dtype is None else options.include_dtype:
        tokens.append(f"{colors.dtype(value.dtype)}")
    tokens.append("sparse " + get_format(value))
    if options.include_shape is not False:
        tokens.append(f"{colors.shape(value.shape)}")
    tokens.append(f"with {instance(value._values).volume} entries:")
    tokens.append(format_summary(value._values, options))
    return " ".join(tokens)


def is_unexpected_dtype(dtype: DType):
    if dtype in [DType(bool), DType(int, 32)]:
        return False
    if dtype.kind == float and dtype.precision == get_precision():
        return False
    return True


def format_tracer(self: Tensor, options: PrintOptions) -> str:
    colors = options.get_colors()
    if self._is_tracer:
        return f"{colors.shape(self.shape)} {colors.dtype(self.dtype)} {colors.value(f'linear tracer for {self.default_backend}')}"
    else:
        return f"{colors.shape(self.shape)} {colors.dtype(self.dtype)} {colors.value(f'{self.default_backend} tracer')}"


def format_full(value: Tensor, options: PrintOptions) -> str:  # multi-line content
    if not value.available:
        return format_tracer(value, options)
    from ._sparse import dense
    value = dense(value)
    import re
    colors = options.get_colors()
    dim_order = tuple(sorted(value.shape.spatial.names, reverse=True))
    lines = []
    formatter = {}
    if options.float_format:
        formatter['float_kind'] = ('{:' + options.float_format + '}').format
    with numpy.printoptions(threshold=np.inf, formatter=formatter):
        if value.shape.dual_rank > 0:  # matrix
            if options.include_shape is not None:
                lines.append(colors.shape(value.shape))
            if value.shape.dual_rank > 1:
                corresponding_primal = value.shape.only(spatial(','.join(dual(value).names)).names, reorder=True)
                if corresponding_primal:
                    value = pack_dims(value, corresponding_primal, corresponding_primal[0].dim_type('&'.join(corresponding_primal.names)))
                value = pack_dims(value, dual, dual('&'.join(value.shape.dual.names)))
            dual_dim = dual(value).name
            primal = dual(value).as_spatial().name
            if primal not in value.shape:
                primal = non_batch(value).non_dual.name
            for b in batch(value).meshgrid(names=True):
                text = " " + np.array2string(value[b].numpy([primal, dual_dim]), separator=', ', max_line_width=np.inf) + " "
                text = re.sub('[\\[\\]]', '', text).replace(',', ' ')
                prefixes = prefix_indices(non_batch(value).non_dual, colors)
                if options.include_shape is not False:
                    for line, prefix in zip(text.split("\n"), prefixes):
                        lines.append(f"{prefix}  {colors.value(line)} along {colors.shape(dual_dim)}")
                else:
                    lines.append(colors.value(text))
        elif value.shape.spatial_rank == 0:  # no spatial or dual dimensions
            if options.include_shape is not None:
                lines.append(colors.shape(value.shape))
            if value.shape.rank <= 1:
                text = np.array2string(value.numpy(), separator=', ', max_line_width=np.inf)
                lines.append(' ' + re.sub('[\\[\\]]', '', text))
            else:
                text = np.array2string(value.numpy(value.shape), separator=', ', max_line_width=np.inf)
                lines.append(text)
        elif value.shape.spatial_rank in (1, 2):
            if value.shape.non_spatial.volume > 1:
                indices = [f"{colors.shape(', '.join(f'{name}={idx}' for name, idx in index_dict.items()))}" for index_dict in value.shape.non_spatial.meshgrid(names=True)]
                max_index_length = max(len(index) for index in indices)
            for i, index_dict in enumerate(value.shape.non_spatial.meshgrid(names=True)):
                row = ""
                if value.shape.non_spatial.volume > 1:
                    row += indices[i] + " " * (max_index_length - len(indices[i]) + 2)
                    if value.shape.spatial_rank == 2:
                        row += "\n"
                if value.shape.spatial_rank == 1:
                    text = np.array2string(value[index_dict].numpy(dim_order), separator=', ', max_line_width=np.inf)
                else:
                    text = " " + np.array2string(value[index_dict].numpy(dim_order)[::-1], separator=', ', max_line_width=np.inf)
                lines.append(row + colors.value(re.sub('[\\[\\]]', '', text)) + (f"  along {colors.shape(spatial(value))}" if options.include_shape is not False else ""))
        else:
            raise NotImplementedError('Can only print tensors with up to 2 spatial dimensions.')
    return "\n".join(lines)


def prefix_indices(index_shape, colors: ColorScheme):
    prefixes = [f"{colors.shape(', '.join(f'{name}={idx}' for name, idx in index_dict.items()))}" for index_dict in index_shape.meshgrid(names=True)]
    max_len = max(len(p) for p in prefixes)
    prefixes = [p + " " * (max_len - len(p) + 2) for p in prefixes]
    return prefixes


def format_row(self: Tensor, options: PrintOptions) -> str:  # all values in a single line
    """
    Including shape:  (x=5, y=4) along vector
    Without shape: (5, 4)
    Auto: don't show if 'vector' but show item names

    Args:
        self:
        options:

    Returns:

    """
    if not self.available:
        return format_tracer(self, options)
    from ._sparse import dense
    self = dense(self)
    colors = options.get_colors()
    if self.shape.rank == 1:
        content = _format_vector(self, options)
        is_vector = self.shape.name == 'vector' and self.shape.channel_rank == 1
        is_dual_vector = self.shape.name == '~vector'
        if (not is_vector and not is_dual_vector) if options.include_shape is None else options.include_shape:
            content += f" along {colors.shape(f'{self.shape.name}{SUPERSCRIPT[self.shape.type]}')}"
        elif is_dual_vector:
            content = "~" + content
    else:
        if channel(self):
            rows = [_format_vector(self[b], options) for b in self.shape.non_channel.meshgrid()]
        else:
            rows = [_format_number(self[b].numpy(), options, self.dtype) for b in self.shape.non_channel.meshgrid()]
        content = "; ".join(rows)
        if options.include_shape is not False:
            content += " " + colors.shape(self.shape)
    if is_unexpected_dtype(self.dtype) if options.include_dtype is None else options.include_dtype:
        content += f" {colors.dtype(self.dtype)}"
    return content


def format_numpy(self: Tensor, options: PrintOptions) -> str:
    from ._sparse import dense
    self = dense(self)
    header = []
    colors = options.get_colors()
    if options.include_shape:
        header.append(colors.shape(self.shape))
    if options.include_dtype:
        header.append(colors.dtype(self.dtype))
    numpy_array = self.numpy(self.shape)
    formatter = {}
    if options.float_format:
        formatter['float_kind'] = ('{:' + options.float_format + '}').format
    with numpy.printoptions(threshold=options.threshold, formatter=formatter):
        content = colors.value(numpy_array)
    return " ".join(header) + "\n" + content if header else content


def _format_vector(self: Tensor, options: PrintOptions) -> str:
    colors = options.get_colors()
    if self.shape.rank > 1:
        from ._magic_ops import flatten
        self = flatten(self, channel('flat'))
    if self.shape.get_item_names(0) is not None and options.include_shape is not False:
        content = ", ".join([f"{item}={_format_number(number, options, self.dtype)}" for number, item in zip(self, self.shape.get_item_names(0))])
    else:
        content = ", ".join([_format_number(num, options, self.dtype) for num in self])
    return colors.value(f"({content})")


def _format_number(num, options: PrintOptions, dtype: DType):
    if options.float_format is not None:
        return format(num, options.float_format)
    if dtype.kind == int:
        return format(num, 'd')
    if dtype.kind == bool:
        return str(bool(num))
    if dtype.kind == float:
        return format(num, options.float_format or '.3f')
    return str(num)


def format_tensor(self: Tensor, options: PrintOptions) -> str:
    if not self.available:
        return format_tracer(self, options)
    if self.shape.is_non_uniform:
        return f"{options.get_colors().shape(self.shape)} non-uniform"
    if options.layout == 'auto':
        if not self.shape:
            return format_summary(self, options)
        if self.shape.volume is not None and self.shape.volume < options.threshold:
            return format_row(self, options)
        else:
            return format_summary(self, options)
    elif options.layout == 'summary':
        return format_summary(self, options)
    elif options.layout == 'full':
        return format_full(self, options)
    elif options.layout == 'row':
        return format_row(self, options)
    elif options.layout == 'numpy':
        return format_numpy(self, options)
    else:
        raise NotImplementedError(f"Layout '{options.layout}' is not supported.")


def is_scalar(value) -> bool:
    """
    Checks whether `value` has no dimensions.

    Args:
        value: `Tensor` or Python primitive or native tensor.

    Returns:
        `bool`
    """
    if isinstance(value, Tensor):
        return value.shape.rank == 0
    elif isinstance(value, numbers.Number):
        return True
    else:
        return len(choose_backend(value).staticshape(value)) == 0


def may_vary_along(value, dims: DimFilter):
    s = value._native_shape if isinstance(value, NativeTensor) else shape(value)
    return s.only(dims).volume > 1


def specs_equal(spec1, spec2):
    if isinstance(spec1, Tensor) or isinstance(spec2, Tensor):
        if isinstance(spec1, Tensor) and isinstance(spec2, Tensor):
            from ._ops import close
            return close(spec1, spec2, rel_tolerance=0, abs_tolerance=0)
        return False
    if isinstance(spec1, dict):
        return set(spec1) == set(spec2) and all([key in spec2 and specs_equal(spec1[key], spec2[key]) for key in spec1.keys()])
    if isinstance(spec1, (tuple, list)):
        return len(spec1) == len(spec2) and all([specs_equal(s1, s2) for s1, s2 in zip(spec1, spec2)])
    return spec1 == spec2
