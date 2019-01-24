from keanu.vertex import UniformInt, Gamma, Poisson, Cauchy, Gaussian
from keanu import BayesNet, KeanuRandom
from keanu.network_io import ProtobufLoader, JsonLoader, ProtobufSaver, DotSaver, JsonSaver
from keanu.vertex import VertexLabel
import pytest


def test_construct_bayes_net() -> None:
    uniform = UniformInt(0, 1)
    graph = set(uniform.get_connected_graph())
    vertex_ids = [vertex.get_id() for vertex in graph]

    assert len(vertex_ids) == 3
    assert uniform.get_id() in vertex_ids

    net = BayesNet(graph)
    latent_vertex_ids = [vertex.get_id() for vertex in net.get_latent_vertices()]

    assert len(latent_vertex_ids) == 1
    assert uniform.get_id() in latent_vertex_ids


@pytest.mark.parametrize("get_method, latent, observed, continuous, discrete",
                         [("get_latent_or_observed_vertices", True, True, True, True),
                          ("get_latent_vertices", True, False, True, True),
                          ("get_observed_vertices", False, True, True, True),
                          ("get_continuous_latent_vertices", True, False, True, False),
                          ("get_discrete_latent_vertices", True, False, False, True)])
def test_can_get_vertices_from_bayes_net(get_method: str, latent: bool, observed: bool, continuous: bool,
                                         discrete: bool) -> None:
    gamma = Gamma(1., 1.)
    gamma.observe(0.5)

    poisson = Poisson(gamma)
    cauchy = Cauchy(gamma, 1.)

    assert gamma.is_observed()
    assert not poisson.is_observed()
    assert not cauchy.is_observed()

    net = BayesNet([gamma, poisson, cauchy])
    vertex_ids = [vertex.get_id() for vertex in getattr(net, get_method)()]

    if observed and continuous:
        assert gamma.get_id() in vertex_ids
    if latent and discrete:
        assert poisson.get_id() in vertex_ids
    if latent and continuous:
        assert cauchy.get_id() in vertex_ids

    assert len(vertex_ids) == (observed and continuous) + (latent and discrete) + (latent and continuous)


def test_probe_for_non_zero_probability_from_bayes_net() -> None:
    gamma = Gamma(1., 1.)
    poisson = Poisson(gamma)

    net = BayesNet([poisson, gamma])

    assert not gamma.has_value()
    assert not poisson.has_value()

    net.probe_for_non_zero_probability(100, KeanuRandom())

    assert gamma.has_value()
    assert poisson.has_value()


def check_loaded_net(net) -> None:
    latents = list(net.get_latent_vertices())
    assert len(latents) == 1
    gamma = latents[0]
    assert gamma.get_value() == 2.5


def check_dot_file(dot_file_name: str) -> None:
    with open(dot_file_name) as f:
        assert len(f.readlines()) == 9


def test_can_save_and_load(tmpdir) -> None:
    PROTO_FILE = str(tmpdir.join("test.proto"))
    JSON_FILE = str(tmpdir.join("test.json"))
    DOT_FILE = str(tmpdir.join("test.dot"))

    gamma = Gamma(1.0, 1.0)
    gamma.set_value(2.5)
    net = BayesNet(gamma.get_connected_graph())
    metadata = {"Team": "GraphOS"}
    protobuf_saver = ProtobufSaver(net)
    protobuf_saver.save(PROTO_FILE, True, metadata)
    json_saver = JsonSaver(net)
    json_saver.save(JSON_FILE, True, metadata)
    dot_saver = DotSaver(net)
    dot_saver.save(DOT_FILE, True, metadata)
    check_dot_file(DOT_FILE)

    protobuf_loader = ProtobufLoader()
    json_loader = JsonLoader()
    new_net_from_proto = protobuf_loader.load(PROTO_FILE)
    check_loaded_net(new_net_from_proto)
    new_net_from_json = json_loader.load(JSON_FILE)
    check_loaded_net(new_net_from_json)


def test_can_get_vertex_by_label() -> None:
    gamma_label = VertexLabel('gamma')
    gaussian_label = VertexLabel('gaussian', ["inner", "outer"])

    gamma = Gamma(1., 1., label=gamma_label)
    gaussian = Gaussian(0., gamma, label=gaussian_label)

    net = BayesNet([gamma, gaussian])
    assert net.get_vertex_by_label(gamma_label) == gamma  # type: ignore
    assert net.get_vertex_by_label('gamma') == gamma  # type: ignore
    assert net.get_vertex_by_label(gaussian_label) == gaussian  # type: ignore


def test_get_vertex_by_label_returns_none_if_not_found() -> None:
    net = BayesNet([Gaussian(0., 1., label="used label"), Gamma(1., 1.)])
    assert net.get_vertex_by_label("unused label") == None  # type: ignore
    assert net.get_vertex_by_label(VertexLabel("used label", ["inner", "outer"])) == None  # type: ignore


def test_can_get_vertices_in_namespace() -> None:
    gamma_label = VertexLabel('gamma', ["inner", "outer"])
    gaussian_label = VertexLabel('gaussian', ["outer"])

    gamma = Gamma(1., 1., label=gamma_label)
    gaussian = Gaussian(0., gamma, label=gaussian_label)

    net = BayesNet([gamma, gaussian])
    vertices_in_outer = list(net.get_vertices_in_namespace(["outer"]))
    vertices_in_inner = list(net.get_vertices_in_namespace(["inner", "outer"]))

    assert len(vertices_in_outer) == 2
    assert gamma in vertices_in_outer
    assert gaussian in vertices_in_outer

    assert len(vertices_in_inner) == 1
    assert gamma in vertices_in_inner


def test_get_vertices_in_namespace_returns_empty_if_not_found() -> None:
    gamma_label = VertexLabel('gamma', ["inner", "outer"])
    gaussian_label = VertexLabel('gaussian', ["inner"])

    gamma = Gamma(1., 1., label=gamma_label)
    gaussian = Gaussian(0., gamma, label=gaussian_label)

    net = BayesNet([gamma, gaussian])
    assert list(net.get_vertices_in_namespace(["outer", "inner"])) == []
