# -*- coding: utf-8 -*-
#***********************************************************************
#* Copyright (c) 2019 Joel Graff <monograff76@gmail.com>               *
#*                                                                     *
#* This program is free software; you can redistribute it and/or modify*
#* it under the terms of the GNU Lesser General Public License (LGPL)  *
#* as published by the Free Software Foundation; either version 2 of   *
#* the License, or (at your option) any later version.                 *
#* for detail see the LICENCE text file.                               *
#*                                                                     *
#* This program is distributed in the hope that it will be useful,     *
#* but WITHOUT ANY WARRANTY; without even the implied warranty of      *
#* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the       *
#* GNU Library General Public License for more details.                *
#*                                                                     *
#* You should have received a copy of the GNU Library General Public   *
#* License along with this program; if not, write to the Free Software *
#* Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307*
#* USA                                                                 *
#*                                                                     *
#***********************************************************************
"""
Utility functions
"""

import math

from collections.abc import Iterable

from operator import sub as op_sub
from operator import add as op_add
from operator import mul as op_mul
from operator import truediv as op_div

from .const import Const

class TupleMath(Const):
    """
    Math support functions for tuples
    """

    @staticmethod
    def _operation(op, op1, op2):
        """
        Generic worker function to perform math operations

        op1, op2 - operands as tuples or lists of tuples

        operation is performed cumulatively on op1 if op2 is None

        where both are defined, operation is performed 1:1 on items, terminating
        at end of lhs list.  If rhs is shorter than lhs, last rhs element is
        is repeated for remainder of lhs
        """

        #early termination for NoneTypes
        if not op1:
            return ()

        #normalize opearnds to lists
        if not isinstance(op1[0], tuple):
            op1 = [op1]

        if op2 is not None:

            if not op2:
                op2 = (0, 0, 0)

            if not isinstance(op2[0], tuple):
                op2 = (op2,)

        #return empty list
        if not op1:
            return op1

        #cumulative operations
        if op2 is None:

            _result = op1[0]

            for _v in op1[1:]:
                _result = tuple((map(op, _result, _v)))

            return _result

        #per-element operations
        _len = len(op2)
        _result = ()
        _last = op2[-1]

        print(op1, op2)

        for _i, _v in enumerate(op1):

            _op2 = _last

            if _i < _len:
                _op2 = op2[_i]

            _result = _result + (tuple(map(op, _v, _op2)),)

        if len(_result) == 1:
            return _result[0]

        return _result

    @staticmethod
    def add(op1, op2=None):
        """
        Add two operands as tuples, lists of tuples, or accumulate
        a single list of tuples (op1)
        """

        return TupleMath._operation(op_add, op1, op2)

    @staticmethod
    def subtract(op1, op2=None):
        """
        Subtract two operands as tuples, lists of tuples, or accumulate
        a single list of tuples (op1)
        """

        return TupleMath._operation(op_sub, op1, op2)

    @staticmethod
    def multiply(op1, op2=None):
        """
        Multiply two operands as tuples, lists of tuples, or accumulate
        a single list of tuples (op1)
        """

        return TupleMath._operation(op_mul, op1, op2)

    @staticmethod
    def divide(op1, op2=None):
        """
        Divide two operands as tuples, lists of tuples, or accumulate
        a single list of tuples (op1)
        """

        return TupleMath._operation(op_div, op1, op2)

    @staticmethod
    def scale(tpl, factor):
        """
        Multiply each component of a tuple or list of tuples by a scalar factor
        """

        _len = len(tpl)

        #get length of longest tuple in list
        if isinstance(tpl[0], tuple):

            _len = 0
            _i = 0

            for _v in tpl:

                _i = len(_v)

                if _i > _len:
                    _len = _i

        return TupleMath._operation(op_mul, tpl, [factor]*_len)

    @staticmethod
    def mean(op1, op2=None):
        """
        Compute the arithmetic mean of two or more tuples
        lhs / rhs - tuples for which mean is to be computed
        if rhs is none / empty, lhs must be an iterable of tuples
        """

        _sum = TupleMath.add(op1, op2)

        #assume cumulative add (op1 = list of tuples, op2 = None)
        _count = len(op1)

        #if op1's first element is a number (not a tuple) and op2 is defined
        #we're averaging two tuples
        if not isinstance(op1[0], Iterable):

            if op2:
                _count = 2

        return TupleMath.scale(_sum, 1.0 / _count)

    @staticmethod
    def length(tpl, ref_point=None):
        """
        Calculate the length of a tuple as a vector from the origin
        If a list of tuples, length is caculated as distance between points
        if reference is defined, all points are calculated from the reference
        """

        _has_ref = ref_point is not None

        #length of a list of points in series or from reference point
        if isinstance(tpl[0], Iterable):

            _len = ()

            _prev = tpl[0:]

            if _has_ref:
                _prev = ref_point
                _len =\
                    (TupleMath.length(TupleMath.subtract(tpl[0], _prev),),)

            for _t in tpl[1:]:

                _len += (TupleMath.length(TupleMath.subtract(_t, _prev)),)

                if not _has_ref:
                    _prev = _t

            if not _len:
                _len = TupleMath.length(_prev)

            return _len

        _tpl = tpl[:]

        if _has_ref:
            _tpl = TupleMath.subtract(tpl, ref_point)[0]

        #vector length from origin to point
        return math.sqrt(sum([_v*_v for _v in _tpl]))

    @staticmethod
    def unit(tpl):
        """
        Normalize a tuple / calculate the unit vector
        """

        _length = TupleMath.length(tpl)

        if not _length:
            return tpl

        return TupleMath.scale(tpl, (1.0 / _length))

    @staticmethod
    def dot(vec1, vec2):
        """
        Calculate the dot product of two tuples
        """

        _sum = 0.0

        for _i, _v in enumerate(vec1):
            _sum += _v * vec2[_i]

        return _sum

    @staticmethod
    def project(vec1, vec2, unit=False):
        """
        Return the projection of vec1 on vec2
        vec1 - Vector to project in tuple form
        vec2 - Vector onto which to project in tuple form
        unit - vec2 is a unit vector
        """

        _vec2_len = 1.0

        if not unit:
            _vec2_len = TupleMath.length(vec2)

        _dot = TupleMath.dot(vec1, vec2)

        return TupleMath.scale(_dot/_vec2_len, vec2)

    @staticmethod
    def bearing(vector, up=(0.0, 1.0)):
        """
        Get the bearing of a vector in tuple form
        """

        if up != (0.0, 1.0):
            up = TupleMath.unit(up[0:2])

        _vec = TupleMath.unit(vector[0:2])

        return math.acos(TupleMath.dot(up, _vec))

    @staticmethod
    def ortho(tpl, is_ccw=True, x_index=0, y_index=1):
        """
        Calculate the orthogonal of x_index and y_index
        """

        _x_sign = -1.0
        _y_sign = 1.0

        if not is_ccw:
            _x_sign, _y_sign = _y_sign, _x_sign

        return (_x_sign * tpl[y_index], _y_sign * tpl[x_index])

    @staticmethod
    def manhattan(lhs, rhs):
        """
        Compute the manhattan distance between two tuples
        Tuples of unequal length are padded with 0.0.
        """

        _distance = 0.0
        _delta = len(rhs) - len(lhs)

        #longer lhs
        if _delta < 0:
            rhs = rhs + (0.0,)*(abs(_delta))

        #longer rhs
        elif _delta > 0:
            lhs = lhs + (0.0,)*(_delta)

        for _i, _v in enumerate(lhs):
            _distance += abs(rhs[_i] - _v)

        return _distance

    @staticmethod
    def cross(src, dest, components=None):
        """
        Calculate the cross-product of two 3D vector tuples
        src - tuple representing starting vector
        dest - tuple representing ending vector
        components - tuple indicating which components to compute.
            Positions define components to cross (y/z, x/z, x/y)
            - 0 = skip; 1 = compute
            - components=(0,0,1) - compute cross of x and y components
        """

        if components is None:
            components = (1.0, 1.0, 1.0)

        assert len(src) == len(dest), """
        TupleMath.cross(): Source and destination vectors of unequal length
        """

        assert len(components) >= len(src), """
        TupleMath.cross(): Components undefined
        """

        _result = [0.0, 0.0, 0.0]
        _idx = (1, 2, 0, 1)

        for _i, _v in enumerate(src):

            if not components[_i]:
                continue

            _a = _idx[_i]
            _b = _idx[_i + 1]

            _result[_i] = src[_a]*dest[_b] - src[_b]*dest[_a]

        return _result

    @staticmethod
    def signed_bearing(src, dest):
        """
        Calculate the signed bearing between two vectors
        """

        _angle = TupleMath.bearing(src, dest)
        _cross = TupleMath.cross(src, dest)
        _dot = TupleMath.dot(_cross, (0.0, 0.0, 1.0))

        if _dot < 0.0:
            _angle *= -1.0

        return _angle

    @staticmethod
    def point_direction(point, vector, epsilon=0.000001):
        """
        Returns: -1 if left, 1 if right, 0 if on line
        """

        _d = TupleMath.cross(vector, point, (0.0, 0.0, 1.0))[2]

        if abs(_d) <= epsilon:
            return 0

        if _d < 0:
            return -1

        return 1
