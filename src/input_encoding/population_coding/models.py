import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from input_encoding.population_coding.process import PopulationCoding


@implements(proc=PopulationCoding, protocol=LoihiProtocol)
@requires(CPU)
class PopulationCodingPyProcModel(PyLoihiProcessModel):
    """Abstract implementation of floating point precision
    leaky-integrate-and-fire neuron model.

    Specific implementations inherit from here.
    """
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out = None  # This will be an OutPort of different LavaPyTypes
    u: np.ndarray = LavaPyType(np.ndarray, float)
    v: np.ndarray = LavaPyType(np.ndarray, float)
    bias: np.ndarray = LavaPyType(np.ndarray, float)
    bias_exp: np.ndarray = LavaPyType(np.ndarray, float)
    du: float = LavaPyType(float, float)
    dv: float = LavaPyType(float, float)
    use_graded_spike: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)