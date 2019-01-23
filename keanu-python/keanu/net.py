from py4j.java_gateway import java_import
from .base import JavaObjectWrapper
from .context import KeanuContext
from .vertex.base import Vertex
from .keanu_random import KeanuRandom
from typing import Any, Iterator, Iterable
from .vartypes import vertex_label_types

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.network.BayesianNetwork")


class BayesNet(JavaObjectWrapper):

    def __init__(self, vertices: Iterable[Any]) -> None:
        java_vertices = k.to_java_object_list(vertices)

        super(BayesNet, self).__init__(k.jvm_view().BayesianNetwork(java_vertices))

    def get_latent_or_observed_vertices(self) -> Iterator[Vertex]:
        return Vertex._to_generator(self.unwrap().getLatentOrObservedVertices())

    def get_latent_vertices(self) -> Iterator[Vertex]:
        return Vertex._to_generator(self.unwrap().getLatentVertices())

    def get_observed_vertices(self) -> Iterator[Vertex]:
        return Vertex._to_generator(self.unwrap().getObservedVertices())

    def get_continuous_latent_vertices(self) -> Iterator[Vertex]:
        return Vertex._to_generator(self.unwrap().getContinuousLatentVertices())

    def get_discrete_latent_vertices(self) -> Iterator[Vertex]:
        return Vertex._to_generator(self.unwrap().getDiscreteLatentVertices())

    def probe_for_non_zero_probability(self, attempts: int, random: KeanuRandom) -> None:
        self.unwrap().probeForNonZeroProbability(attempts, random.unwrap())

    def get_vertex_by_label(self, label: vertex_label_types) -> Vertex:
        return Vertex(self.unwrap().getVertexByLabel(label.unwrap()))

    def get_vertices_in_namespace(self, namespace: List[str]) -> Iterator[Vertex]:
        return Vertex._to_generator(self.unwrap().getVerticesInNamespace(k.to_java_string_array(namespace)))
