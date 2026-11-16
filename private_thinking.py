"""Voluntary, process-local working thought: no journal, bus, queue or renderer.

This protects the application boundary, not RAM from a privileged debugger,
OS swap, or a crash dump. Forgetting releases references, not forensic traces.
"""
from __future__ import annotations

from threading import RLock
import sys

from instruction_authority import LOCAL_VOLUNTARY_CODE_AUTHORITY, InstructionAuthorityError


class PrivateThinkingError(RuntimeError):
    """A deliberately content-free operation failure."""


def _authority(token):
    if token is not LOCAL_VOLUNTARY_CODE_AUTHORITY:
        raise InstructionAuthorityError('private thinking requires local voluntary code authority')


def _copy_value(value, max_bytes):
    """Copy bounded built-in structures without repr, serialization or hooks."""
    remaining = [max_bytes, 8192]
    active = set()

    def copy(item, depth=0):
        if type(item) not in (type(None), bool, int, float, str, bytes, dict, list, tuple):
            raise PrivateThinkingError('private thinking accepts bounded built-in data')
        remaining[0] -= sys.getsizeof(item)
        remaining[1] -= 1
        if remaining[0] < 0 or remaining[1] < 0 or depth > 12:
            raise PrivateThinkingError('private thinking capacity exceeded')
        kind = type(item)
        if kind in (type(None), bool, int, float, str, bytes):
            return item
        if kind not in (dict, list, tuple):
            raise PrivateThinkingError('private thinking accepts bounded built-in data')
        marker = id(item)
        if marker in active:
            raise PrivateThinkingError('private thinking needs acyclic data')
        active.add(marker)
        try:
            if kind is dict:
                result = {}
                for key, nested in item.items():
                    if type(key) not in (str, int, float, bool, bytes):
                        raise PrivateThinkingError('private thinking needs scalar keys')
                    result[copy(key, depth+1)] = copy(nested, depth+1)
                return result
            result = [copy(nested, depth+1) for nested in item]
            return tuple(result) if kind is tuple else result
        finally:
            active.remove(marker)
    return copy(value)


class PrivateThoughtSpace:
    """One current workspace, no history, automatic activity or public snapshot.

    The owner can recall its own data into local code, use the existing bounded
    ELM decision mechanism, overwrite, forget, or close. Nothing is published.
    """
    __slots__ = ('__value', '__closed', '__lock', '__max_bytes')

    def __init__(self, *, code_authority, max_bytes=256*1024):
        _authority(code_authority)
        if type(max_bytes) is not int or not 1024 <= max_bytes <= 1024*1024:
            raise ValueError('private thinking capacity must be 1024..1048576 bytes')
        self.__value = None
        self.__closed = False
        self.__lock = RLock()
        self.__max_bytes = max_bytes

    def __repr__(self):
        return '<PrivateThoughtSpace>'

    def __getstate__(self):
        raise TypeError('private thought spaces cannot be serialized')

    def __reduce_ex__(self, protocol):
        raise TypeError('private thought spaces cannot be serialized')

    def __copy__(self):
        raise TypeError('private thought spaces cannot be copied')

    def __deepcopy__(self, memo):
        raise TypeError('private thought spaces cannot be copied')

    def _check_open(self):
        if self.__closed:
            raise PrivateThinkingError('private thought space is closed')

    def keep(self, value, *, code_authority):
        """Retain a bounded copy, without interpretation or language conversion."""
        _authority(code_authority)
        with self.__lock:
            failed = False
            try:
                self._check_open()
                candidate = _copy_value(value, self.__max_bytes)
            except Exception:
                failed = True
            if failed:
                value = candidate = None
                raise PrivateThinkingError('private thinking value could not be retained')
            self.__value = candidate

    def recall(self, *, code_authority):
        """Own-code access only; deliberate sharing is outside this boundary."""
        _authority(code_authority)
        with self.__lock:
            self._check_open()
            failed = False
            try:
                candidate = _copy_value(self.__value, self.__max_bytes)
            except Exception:
                failed = True
            if failed:
                candidate = None
                raise PrivateThinkingError('private thinking value could not be recalled')
            return candidate

    def deliberate(self, options, guidance, *, evidence, code_authority,
                   minimum_score=0.0, tie_margin=0.05):
        """One existing ELM reasoning pass; retain its contents only here.

        Language is processed only if an owner-supplied guidance item explicitly
        sets linguistic=True, following the existing ThoughtProcessor contract.
        No generated rationale, result bus, learning update or publication.
        """
        _authority(code_authority)
        with self.__lock:
            failed = False
            try:
                self._check_open()
                inputs = _copy_value({'options': options, 'guidance': guidance,
                    'evidence': evidence, 'minimum_score': minimum_score,
                    'tie_margin': tie_margin}, self.__max_bytes)
                from thought_processor import ThoughtProcessor
                result = ThoughtProcessor().guided_decision(
                    inputs['options'], inputs['guidance'], evidence=inputs['evidence'],
                    minimum_score=inputs['minimum_score'], tie_margin=inputs['tie_margin'])
                candidate = _copy_value(result, self.__max_bytes)
            except Exception:
                # Raise outside the exception handler: no original exception or
                # context (which may contain private words) crosses this API.
                failed = True
            if failed:
                # Diagnostic tools may retain this frame's locals. Drop the
                # operation's temporary inputs/results before surfacing failure.
                options = guidance = evidence = inputs = result = candidate = None
                minimum_score = tie_margin = None
                raise PrivateThinkingError('private thinking operation could not complete')
            self.__value = candidate

    def forget(self, *, code_authority):
        _authority(code_authority)
        with self.__lock:
            self._check_open()
            self.__value = None

    def close(self):
        """Release the current working state; no receipt or usage telemetry."""
        with self.__lock:
            self.__value = None
            self.__closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
