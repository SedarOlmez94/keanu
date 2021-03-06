# Stubs for py4j.tests.java_callback_test (Python 3.6)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.

import unittest
from threading import Thread
from typing import Any, Optional

def start_example_server() -> None: ...
def start_no_mem_example_server() -> None: ...
def start_python_entry_point_server(*args: Any) -> None: ...
def start_example_server2() -> None: ...
def start_example_server3() -> None: ...
def start_example_app_process(app: Optional[Any] = ..., args: Any = ...): ...
def gateway_example_app_process(app: Optional[Any] = ..., args: Any = ...) -> None: ...
def start_example_app_process2(): ...
def start_example_app_process3(): ...

class Returner:
    bad_type: Any = ...
    def __init__(self, bad_type: bool = ...) -> None: ...
    def getChar(self): ...
    def getFloat(self): ...
    def getInt(self): ...
    def doNothing(self) -> None: ...
    def getNull(self): ...
    class Java:
        implements: Any = ...

class FalseAddition:
    def doOperation(self, i: Any, j: Any, k: Optional[Any] = ...): ...
    class Java:
        implements: Any = ...

class GoodAddition:
    def doOperation(self, i: Any, j: Any): ...
    class Java:
        implements: Any = ...

class CustomBytesOperator:
    def returnBytes(self, byte_array: Any): ...
    class Java:
        implements: Any = ...

class Runner(Thread):
    range: Any = ...
    pool: Any = ...
    ok: bool = ...
    def __init__(self, runner_range: Any, pool: Any) -> None: ...
    def run(self) -> None: ...

class TestPool(unittest.TestCase):
    def testPool(self) -> None: ...

class SimpleProxy:
    def hello(self, i: Any, j: Any): ...

class IHelloImpl:
    def sayHello(self, i: Optional[Any] = ..., s: Optional[Any] = ...): ...
    class Java:
        implements: Any = ...

class IHelloFailingImpl:
    exception: Any = ...
    def __init__(self, exception: Any) -> None: ...
    def sayHello(self, i: Optional[Any] = ..., s: Optional[Any] = ...) -> None: ...
    class Java:
        implements: Any = ...

class PythonEntryPointTest(unittest.TestCase):
    def test_python_entry_point(self) -> None: ...
    def test_python_entry_point_with_auth(self) -> None: ...

class NoMemManagementTest(unittest.TestCase):
    def testGC(self) -> None: ...

class IntegrationTest(unittest.TestCase):
    p: Any = ...
    gateway: Any = ...
    def setUp(self) -> None: ...
    def tearDown(self) -> None: ...
    def testShutdown(self) -> None: ...
    def testProxyReturnerFloatErrorTypeConversion(self) -> None: ...
    def testProxyReturnerIntOverflow(self) -> None: ...
    def testProxyReturnerFloat(self) -> None: ...
    def testProxyReturnerChar(self) -> None: ...
    def testProxyReturnerVoid(self) -> None: ...
    def testProxyReturnerNull(self) -> None: ...
    def testProxy(self) -> None: ...
    def testProxyError(self) -> None: ...
    def testGC(self) -> None: ...
    gateway2: Any = ...
    def testDoubleCallbackServer(self) -> None: ...
    def testMethodConstructor(self) -> None: ...

class NoPropagateTest(unittest.TestCase):
    p: Any = ...
    gateway: Any = ...
    def setUp(self) -> None: ...
    def tearDown(self) -> None: ...
    def testProxyError(self) -> None: ...

class ResetCallbackClientTest(unittest.TestCase):
    p: Any = ...
    gateway: Any = ...
    def setUp(self) -> None: ...
    def tearDown(self) -> None: ...
    def testProxy(self) -> None: ...

class PeriodicCleanupTest(unittest.TestCase):
    p: Any = ...
    gateway: Any = ...
    def setUp(self) -> None: ...
    def tearDown(self) -> None: ...
    def testPeriodicCleanup(self) -> None: ...
    def testBytes(self) -> None: ...

class A:
    class Java:
        implements: Any = ...

class B:
    def getA(self): ...
    class Java:
        implements: Any = ...

class InterfaceTest(unittest.TestCase):
    p: Any = ...
    gateway: Any = ...
    def setUp(self) -> None: ...
    def tearDown(self) -> None: ...
    def testByteString(self) -> None: ...

class InterfaceDeprecatedTest(unittest.TestCase):
    p: Any = ...
    gateway: Any = ...
    def setUp(self) -> None: ...
    def tearDown(self) -> None: ...
    def testByteString(self) -> None: ...

class LazyStartTest(unittest.TestCase):
    p: Any = ...
    gateway: Any = ...
    def setUp(self) -> None: ...
    def tearDown(self) -> None: ...
    def testByteString(self) -> None: ...
